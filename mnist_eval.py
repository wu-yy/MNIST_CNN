import time
import  tensorflow as tf

from tensorflow.examples.tutorials.mnist import  input_data
#加载mnist_inferenc.py 和 mnist_train.py 里面的常量和函数
import mnist_inference
import mnist_train
import numpy as np
BATCH_SIZE=100
#每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出格式
        x = tf.placeholder(tf.float32, [1000, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS], name='x-input')

        y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name='y-input')
        reshaped_xs = np.reshape(mnist.validation.images[0:1000], (1000, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS))
        validate_feed={x:reshaped_xs,y_:mnist.validation.labels[0:1000]}

        y=mnist_inference.inference(input_tensor=x,train=None,regularizer=None)

        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variable_to_restore)

        #每隔一段时间检测正确率的变化
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("在第%s 次训练之后，验证数据的正确率:%g"%(global_step,accuracy_score))
                else:
                    print("No check point find")
                    return

            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("./data/MNIST/", one_hot=True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()
