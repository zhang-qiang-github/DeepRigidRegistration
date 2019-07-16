import tensorflow as tf
import os
import sklearn
import scipy.io as sio
import numpy as np
from PIL import Image

def data_img(path, batch):
    filenames = os.listdir(path)
    while True:
        for start in range(0, len(filenames),batch):
            batch_x = []
            batch_y = []
            angle = []
            end = min(start + batch, len(filenames))
            if start+batch >= len(filenames):
                continue
            train_batch = filenames[start:end]
            for filename in train_batch:
                tof_path = os.path.join(path,filename)
                img = Image.open(tof_path)
                img = img.resize((320,320))
                ori_img = np.array(img)[:,:,0]
                img = Image.fromarray(ori_img)
                a = np.random.random(1)[0]
                angle.append(a)
                img = img.rotate(a*180)
                np_img = np.array(img).astype(np.float32)
                IMG = np.zeros(shape=[320,320,2])
                IMG[:,:,0] = ori_img
                IMG[:,:,1] = np_img
                batch_x.append(IMG)
            batch_x = np.array(batch_x)
            angle = np.array(angle)
            yield batch_x, angle

def weight_variable(shape,name=''):
  initial = tf.truncated_normal(shape, stddev=0.01)
  w = tf.get_variable(name='weight_'+name,initializer=initial)
  tf.summary.histogram(w._shared_name, w)
  return w

def bias_variable(shape,name=''):
  initial = tf.constant(0.1, shape=shape)
  b = tf.get_variable(name='bias_'+name, initializer=initial)
  tf.summary.histogram(b._shared_name, b)
  return b

def one_conv_layer(input_tensor, kernal, output_channal, strides=(1,1),name='',padding='SAME'):
    input_channal = int(input_tensor.get_shape()[-1])
    with tf.variable_scope("one_conv_layer_"+name, reuse=tf.AUTO_REUSE):
        w = weight_variable([kernal[0], kernal[1], input_channal, output_channal],name=name)  # [kernelsize1, kernelsize2, input channal, output channal]
        b = bias_variable([output_channal],name=name)
        x = tf.nn.conv2d(input_tensor, w, strides=[1, strides[0], strides[1], 1], padding=padding) + b
    return x

input_img = tf.placeholder(dtype=tf.float32, shape=[None,320,320,2])
angle_tensor = tf.placeholder(dtype=tf.float32, shape=[None])

x = one_conv_layer(input_img, kernal=[3,3], output_channal=32, name='a1')
x = one_conv_layer(x, kernal=[3,3], output_channal=32, name='a2')
x = tf.layers.batch_normalization(x, axis=3, training=True)
pool1 = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = one_conv_layer(pool1, kernal=[3,3], output_channal=64, name='b1')
x = one_conv_layer(x, kernal=[3,3], output_channal=64, name='b2')
x = tf.layers.batch_normalization(x, axis=3, training=True)
pool2 = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = one_conv_layer(pool2, kernal=[3,3], output_channal=128, name='c1')
x = one_conv_layer(x, kernal=[3,3], output_channal=128, name='c2')
x = tf.layers.batch_normalization(x, axis=3, training=True)
pool3 = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

x = one_conv_layer(pool3, kernal=[3,3], output_channal=256, name='d1')
x = one_conv_layer(x, kernal=[3,3], output_channal=256, name='d2')
x = one_conv_layer(x, kernal=[3,3], output_channal=256, name='d2')
x = tf.layers.batch_normalization(x, axis=3, training=True)
pool4 = tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

s = tf.layers.flatten(pool4)
N = int(s.shape[1])

w1 = weight_variable(shape=[N,256],name='flatten1')
b1 = bias_variable(shape=[256],name='flatten1')
s1 = tf.matmul(s,w1)+b1
s2 = tf.layers.batch_normalization(s1,axis=1,training=True)
s3 = tf.nn.relu(s2)

w2 = weight_variable(shape=[256,64],name='flatten2')
b2 = bias_variable(shape=[64],name='flatten2')
s4 = tf.matmul(s3,w2)+b2
s5 = tf.layers.batch_normalization(s4,axis=1,training=True)
s6 = tf.nn.relu(s5)

w3 = weight_variable(shape=[64,1],name='flatten3')
b3 = bias_variable(shape=[1],name='flatten3')
para = tf.matmul(s6,w3)+b3

loss = tf.reduce_mean(tf.square(para[:,0]-angle_tensor))
error = tf.reduce_mean(tf.abs(para[:,0]-angle_tensor)*180)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

from stn import spatial_transformer_network as transformer
pred_image = []
for i in range(16):
    theta = tf.stack([(tf.cos(para[i,0]), -tf.sin(para[i,0]), tf.constant(0.0)), (tf.sin(para[i,0]), tf.cos(para[i,0]), tf.constant(0.0))], axis=0)
    iImg = input_img[i,:,:,0]
    iImg = tf.expand_dims(iImg, axis=0)
    iImg = tf.expand_dims(iImg, axis=3)
    pImg = transformer(iImg, theta, out_dims=[320,320])
    pred_image.append(pImg)
pred_image = tf.concat(pred_image, axis=0)

optimizer = tf.train.AdamOptimizer(1e-3)

# train = optimizer.minimize(loss)

train_op = optimizer.minimize(loss)
# train_op = tf.group([train_op, update_ops])
with tf.control_dependencies(update_ops):
   train_op = optimizer.minimize(loss)

init = tf.initialize_all_variables()



sess = tf.Session()
sess.run(init)

data_generator = data_img('./data',8)

for epoch in range(100):
    for i in range(500):

        input_image, angle = data_generator.__next__()
        _,l, e = sess.run([train_op, loss, error],feed_dict={input_img:input_image, angle_tensor:angle})
        # p = para*180/3.1415926
        if i == 100:
            p = sess.run(para, feed_dict={input_img: input_image, angle_tensor: angle})

        print("iteration: %d  || loss = %.3f  || error (degree) = %.3f"%(epoch*500+i, l, e))

