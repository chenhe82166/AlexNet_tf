from importData import Dataset
testing = Dataset('wxb_pic/pic_test', '.jpg')

import tensorflow as tf
import numpy as np

# Parameters
batch_size = 1

ckpt = tf.train.get_checkpoint_state("save")
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')# 恢复tensorflow图，也就是读取神经网络的结构，从而无需再次构建网络
pred = tf.get_collection("pred")[0]# tf.get_collection：从一个结合中取出全部变量，是一个列表
x = tf.get_collection("x")[0]
keep_prob = tf.get_collection("keep_prob")[0]

# Launch the graph
# with tf.Session() as sess:
sess = tf.Session()
saver.restore(sess, ckpt.model_checkpoint_path)# 测试阶段使用saver.restore()方法恢复变量：
                                               # sess：表示当前会话，之前保存的结果将被加载入这个会话
                                               # ckpt.model_checkpoint_path：表示模型存储的位置，不需要提供模型的名字，它会去查看checkpoint文件，看看最新的是谁，叫做什么。
  
# test
step_test = 1
while step_test * batch_size < len(testing):
    testing_ys, testing_xs = testing.nextBatch(batch_size)
    predict = sess.run(pred, feed_dict={x: testing_xs, keep_prob: 1.})
    print "Testing label:", testing.label2category[np.argmax(testing_ys, 1)[0]]
    print "Testing predict:", testing.label2category[np.argmax(predict, 1)[0]]
    step_test += 1
