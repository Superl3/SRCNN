import tensorflow as tf
import numpy as np
import random
from PIL import Image
import os

width = 32
height = 32

batch_size = 128
epoch = 5000

learning_rate = 1e-4
stddev = 5e-2
ratio = 0.7

def load_images(paths):
    imgs = []
    for path in paths:
        for filename in os.listdir(path):
            img = Image.open(os.path.join(path, filename)).convert('L')
            imgs.append(img)
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

# CNN 모델
def CNN(input_size, x, stddev):

    Wzh = tf.Variable(tf.truncated_normal(shape=[3,3,2,32],stddev=stddev))
    
    Whh = tf.Variable(tf.truncated_normal(shape=[3,3,32,32],stddev=stddev))
    bh = tf.Variable(tf.constant(0.1, shape=[32]))
    
    Why = tf.Variable(tf.truncated_normal(shape=[3,3,32,1],stddev=stddev))
    by = tf.Variable(tf.constant(0.1, shape=[1]))

    h0 = tf.zeros([input_size, 32, 32, 32])

    z0 = tf.concat([x,x], axis=3)
    h1 = tf.nn.relu(tf.nn.conv2d(h0, Whh, strides=[1,1,1,1], padding='SAME') + tf.nn.conv2d(z0, Wzh, strides=[1,1,1,1], padding='SAME') + bh)
    y1 = tf.nn.relu(tf.nn.conv2d(h1, Why, strides=[1,1,1,1], padding='SAME') + by)

    z1 = tf.concat([x,y1], axis=3)
    h2 = tf.nn.relu(tf.nn.conv2d(h1, Whh, strides=[1,1,1,1], padding='SAME') + tf.nn.conv2d(z1, Wzh, strides=[1,1,1,1], padding='SAME') + bh)
    y2 = tf.nn.relu(tf.nn.conv2d(h2, Why, strides=[1,1,1,1], padding='SAME') + by)

    z2 = tf.concat([x,y2], axis=3)
    h3 = tf.nn.relu(tf.nn.conv2d(h2, Whh, strides=[1,1,1,1], padding='SAME') + tf.nn.conv2d(z2, Wzh, strides=[1,1,1,1], padding='SAME') + bh)
    y3 = tf.clip_by_value(tf.nn.relu(tf.nn.conv2d(h3, Why, strides=[1,1,1,1], padding='SAME') + by),0,1)

    saver = tf.train.Saver([Wzh,Whh,bh,Why,by])

    return y1, y2, y3, saver

def do_CNN(batch_size, stddev, sets, ratio, width, height): 

    tf.logging.set_verbosity(tf.logging.ERROR)
    
    x = tf.placeholder(tf.float32, shape=[None, 32,32,1])
    y = tf.placeholder(tf.float32, shape=[None, 32,32,1])
    input_size = tf.placeholder(tf.float32)
    
    y1, y2, y3, saver = CNN(input_size, x, 5e-2)
    loss = tf.reduce_mean(tf.squared_difference(y, y1) + tf.squared_difference(y, y2) + tf.squared_difference(y, y3))
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    psnr = tf.reduce_mean(tf.image.psnr(y, y3, max_val=1.0))

    tf.summary.image("image_hypo", y3, max_outputs=5)
    tf.summary.image("image_y", y, max_outputs=5)
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("psnr", psnr)

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
                psnr_, loss_, summary = sess.run([psnr, loss, merge], feed_dict={x: test_x, y: test_y, input_size: len(test_x)})
                print("Epoch : %4d, psnr : %.4fdB, loss : %.4f" %(i, psnr_, loss_))
                writer.add_summary(summary, global_step)
                global_step+=1
                saver.save(sess, 'RNNParams')

            sess.run(train, feed_dict={x: train_x, y: train_y, input_size: len(train_x)})

do_CNN(batch_size, stddev, load_images(['291/']), ratio, width, height)