import  tensorflow as tf

#配置神经网络的参数
INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28 #图片的长和宽 28
NUM_CHANNELS=1  #RGB通道为1 ，单色图片
NUM_LABELS=10   #图片的种类 10

#第一层卷基层的尺寸和深度
CONV1_SIZE=5
CONV1_DEEP=32

#第二层卷基层的尺寸和宽度
CONV2_SIZE=5
CONV2_DEEP=64

#全连接层的节点个数
FC_SIZE=512

#定义卷积神经网络的向前传播过程，这里添加一个参数train ，用于区分训练过程和测试过程。在程序中将用到dropout方法
#dropout可以进一步提升模型的可靠性，防止过度拟合，dropout只在训练的时候使用

def inference(input_tensor,train,regularizer):

    #定义了输入为28*28*1的原始图片像素，因为卷积层中使用了全0填充，所以输出为 28*28*32de 矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights=tf.get_variable("weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        con1_biases=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))

        #使用边长为5，深度为32的过滤器，过滤器的步长为1，且使用全0填充
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,con1_biases))

    #实现第二层池话层的前向传播过程。这里选用最大池化层，池化层的边长为2，使用全0填充，也移动的步长为2。这一层的输入是上一层的输出，也就是28*28*32的矩阵
    #输出是14*14*32的矩阵
    with tf.name_scope('layer1-pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #声明第三层卷积层的变量并实现前向传播过程，这一层的输入是14*14*32的矩阵
    #输出是14*14*64的矩阵
    with tf.variable_scope('layer2-conv2'):
        conv2_weights=tf.get_variable("weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable('bias',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))

        #使用边长为5，深度为64的过滤器，过滤器的移动补偿为1，使用全0填充
        conv2=tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    #实现第4层的池化层的前向传播过程，这一层的输入为14*14*64的矩阵
    #输出是7*7*64的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    #将第四层的输出层转化为第五层的全连接层。第四层的输出是7*7*64的矩阵
    #将7*7*64的矩阵拉成一个向量。
    #pool2.get_shape 函数可以得到第四层的维度，而不需手工计算。因为每一层神经网络的输入输出都为一个batch矩阵，所以这里得到的维度
    #也包含了一个batch中数据的个数

    pool_shape=pool2.get_shape().as_list()

    #计算将矩阵拉成向量之后的长度，这个长度就是矩阵的长宽以及深度的成绩。注意pool_shape[0] 为一个batch中数据的个数
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

    #通过tf.reshape 函数将第四层的输出变成一个batch向量
    reshapeed=tf.reshape(pool2,[pool_shape[0],nodes])

    #声明第五层的全连接变懒并实现前向传播过程。这一层的输入是拉直之后的一组向量，向量的长度为3136
    #输出是一组长度为512的向量。这一层引入了dropout,dropout在训练的会随机将部分的输出的改为0，一般只在全连接层而不是卷积层或者池化层使用。

    with tf.variable_scope('layer5-fc1'):
        fc1_weights=tf.get_variable('weight',[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))

        #只有在全连接层的权重需要加入正则化。
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases=tf.get_variable('bias',[FC_SIZE],initializer=tf.constant_initializer(0.1))

        fc1=tf.nn.relu(tf.matmul(reshapeed,fc1_weights)+fc1_biases)
        if train:
            fc1=tf.nn.dropout(fc1,0.5)
    #声明第6层全连接层的变量实现前向传播过程，这一层的输入是512的向量，输出是一组长度为10的向量，这一层的输出通过soft_max之后
    #就得到了最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights=tf.get_variable('weight',[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_weights))
        fc2_biases=tf.get_variable('bias',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,fc2_weights)+fc2_biases

    #返回第六层的输出
    return logit
