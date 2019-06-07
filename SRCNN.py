import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import os

width = 32
height = 32
learning_rate = 1e-3
batch_size = 128
epoch = 10000
stddev = 5e-2
ratio = 0.7

def load_images(paths):
    imgs = []
    for path in paths:
        for filename in os.listdir(path):
            imgs.append(Image.open(os.path.join(path, filename)).convert('L'))
    return imgs

def div_sets(batch_size, ratio, imgs):

    num = len(imgs)
    idx = np.arange(0, num)
    np.random.shuffle(idx)

    train = []
    test = []
    
    for i in range(0,int(num*ratio)):
        if i >= batch_size:
            break
        train.append(imgs[idx[i]])
    for i in range(int(num*ratio)+1, num-1):
        if i >= int(num*ratio)+1 + batch_size:
            break
        test.append(imgs[idx[i]])
    
    return train, test

def random_crop(imgs, width, height):
    
    sets = []
    for img in imgs:    
        sizeX, sizeY = img.size
        maxx = int(random.random() * sizeX) - (width + 1)
        if maxx < 0:
            maxx = 1
        maxy = int(random.random() * sizeY) - (height + 1)
        if maxy < 0:
            maxy = 1
        sets.append(img.crop((maxx, maxy, maxx + width, maxy + height)))
    return sets

def create_sets(imgs):

    raw = []
    low = []
    width, height = imgs[0].size

    for img in imgs:
        
        raw_res = np.array(img).reshape(width,height,1)/255
        low_res = np.array(img.resize((int(width/2),int(height/2))).resize((width,height))).reshape(width,height,1)/255

        raw.append(raw_res)
        low.append(low_res)

    return low, raw


def do_CNN(batch_size, stddev, sets, ratio, width, height): 

    tf.logging.set_verbosity(tf.logging.ERROR)

    x = tf.placeholder(tf.float32, shape=[None, 32,32,1])
    y = tf.placeholder(tf.float32, shape=[None, 32,32,1])

    W1 = tf.Variable(tf.truncated_normal(shape=[3,3, 1,64],stddev=stddev))
    W2 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev=stddev))
    W3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,1],stddev=stddev))

    b1 = tf.Variable(tf.constant(0.1, shape=[64]))
    b2 = tf.Variable(tf.constant(0.1, shape=[64]))
    b3 = tf.Variable(tf.constant(0.1, shape=[1]))
    
    Layer1 = tf.nn.relu(tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME') + b1)    
    Layer2 = tf.nn.relu(tf.nn.conv2d(Layer1, W2, strides=[1,1,1,1], padding='SAME') + b2)
    hypothesis = tf.nn.conv2d(Layer2, W3, strides=[1,1,1,1], padding='SAME') + b3

    loss = tf.reduce_mean(tf.squared_difference(y,hypothesis))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    psnr = tf.reduce_mean(tf.image.psnr(y, hypothesis, max_val=1.0))
    
    tf.summary.image("image_hypo", hypothesis, max_outputs=5)
    tf.summary.image("image_y", y, max_outputs=5)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("psnr", psnr)

    saver = tf.train.Saver([W1,b1,W2,b2,W3,b3])

    global_step = 0

    with tf.Session() as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./logs/", sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(epoch):
            train_raw, test_raw = div_sets(batch_size, ratio, sets)
            train_x, train_y = create_sets(random_crop(train_raw, width, height))
   
            if i%100 == 0:
                test_x, test_y = create_sets(random_crop(test_raw, width, height))
                psnr_, loss_, summary = sess.run([psnr, loss, merge], feed_dict={x: test_x, y: test_y})
                print("Epoch : %4d, psnr : %.4fdB, loss : %.4f" %(i, psnr_, loss_))
                writer.add_summary(summary, global_step)
                global_step+=1
                saver.save(sess, 'SRCNNParams')

            sess.run(train, feed_dict={x: train_x, y: train_y})    

do_CNN(batch_size, stddev, load_images(['291/']), ratio, width, height)