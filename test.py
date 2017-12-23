from tensorflow.examples.tutorials.mnist import  input_data

#从路径中导入MNIST数据集
mnist=input_data.read_data_sets("./data/MNIST/",one_hot=True)

#打印训练数据集的大小
print("训练数据集的大小是:",mnist.train.num_examples)

print("验证数据集的大小是：",mnist.validation.num_examples)

print("测试数据集的大小是：",mnist.test.num_examples)
print(len(mnist.validation.images))