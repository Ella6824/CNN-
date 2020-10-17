import time

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model


# =======================================================================
# 获取一张图片
from preHandle import get_files


def get_one_image(train):
    # 输入参数：train,训练图片的路径
    # 返回参数：image，从训练图片中随机抽取一张图片
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]  # 随机选择测试的图片
    target_img_dir = './results/'
    save_img(img_dir, target_img_dir)
    img = Image.open(img_dir)
    image = np.array(img)
    return image


def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 4

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 256, 256, 3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[256, 256, 3])

        # you need to change the directories to yours.
        logs_train_dir = './logs'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('This is a husky with possibility %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('This is a jiwawa with possibility %.6f' % prediction[:, 1])


def save_img(img_dir, target_img_dir):
    img = Image.open(img_dir)
    plt.xticks([])  # 去掉横坐标轴
    plt.yticks([])  # 去掉纵坐标轴
    plt.imshow(img)
    plt.savefig(target_img_dir + str(int(time.time())) + '.jpg')
    plt.show()

if __name__ == '__main__':
    train_dir = './train_data_resized'
    train, train_label, val, val_label = get_files(train_dir, 0.1)
    img = get_one_image(val)  # 通过改变参数train or val，进而验证训练集或测试集
    evaluate_one_image(img)