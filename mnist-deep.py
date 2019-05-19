{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\python\\framework\\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Colocations handled automatically by placer.\n",
      "60100\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "a = tf.constant(100, name='a')\n",
    "b = tf.constant(200, name='b')\n",
    "c = tf.constant(300, name='c')\n",
    "v = tf.Variable(0, name='v')\n",
    "\n",
    "calc_op = a + b * c\n",
    "assign_op = tf.assign(v, calc_op)\n",
    "\n",
    "sess = tf.Session()\n",
    "\n",
    "tw = tf.summary.FileWriter('log_dir', graph=sess.graph)\n",
    "\n",
    "sess.run(assign_op)\n",
    "print(sess.run(v))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "60100\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "a = tf.constant(100, name='a')\n",
    "b = tf.constant(200, name='b')\n",
    "c = tf.constant(300, name='c')\n",
    "v = tf.Variable(0, name='v')\n",
    "\n",
    "calc_op = a + b * c\n",
    "assign_op = tf.assign(v, calc_op)\n",
    "\n",
    "sess = tf.Session()\n",
    "\n",
    "tw = tf.summary.FileWriter('log_dir', graph=sess.graph)\n",
    "\n",
    "sess.run(assign_op)\n",
    "print(sess.run(v))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-3-a8a244d9335a>:2: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n",
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please write your own downloading logic.\n",
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use urllib or similar directly.\n",
      "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.\n",
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting mnist/train-images-idx3-ubyte.gz\n",
      "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.\n",
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting mnist/train-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.one_hot on tensors.\n",
      "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.\n",
      "Extracting mnist/t10k-images-idx3-ubyte.gz\n",
      "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.\n",
      "Extracting mnist/t10k-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From C:\\Users\\Jaeil\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"mnist/\", one_hot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.\n",
      "Extracting minst/train-images-idx3-ubyte.gz\n",
      "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.\n",
      "Extracting minst/train-labels-idx1-ubyte.gz\n",
      "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.\n",
      "Extracting minst/t10k-images-idx3-ubyte.gz\n",
      "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.\n",
      "Extracting minst/t10k-labels-idx1-ubyte.gz\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "\n",
    "#MNIST 손글씨 이미지 데이터 읽어 들이기\n",
    "mnist = input_data.read_data_sets(\"minst/\", one_hot=True)\n",
    "\n",
    "pixels = 28*28\n",
    "nums = 10 # 0-9 사이의 카테고리"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#플레이스홀더 정의하기\n",
    "x = tf.placeholder(tf.float32, shape=(None, pixels), name=\"x\") #이미지 데이터\n",
    "y_ = tf.placeholder(tf.float32, shape=(None, nums), name=\"y_\") #정답 레이블"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#가중치와 바이어스를 초기화하는 함수\n",
    "def weight_variable(name, shape):\n",
    "    W_init = tf.truncated_normal(shape, stddev=0.1)\n",
    "    W = tf.Variable(W_init, name=\"W_\"+name)\n",
    "    return W\n",
    "\n",
    "def bias_variable(name, size):\n",
    "    b_init = tf.constant(0.1, shape=[size])\n",
    "    b = tf.Variable(b_init, name=\"b_\"+name)\n",
    "    return b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#합성공 계층을 만드는 함수\n",
    "def conv2d(x, W):\n",
    "    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#최대 풀링츠을 만드는 함수\n",
    "def max_pool(x):\n",
    "    return tf.nn.max_pool(x, ksize=[1,2,2,1],\n",
    "        strides=[1,2,2,1], padding='SAME')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#합성곱층1\n",
    "with tf.name_scope('conv1')as scope:\n",
    "    W_conv1 = weight_variable('conv1', [5,5,1,32])\n",
    "    b_conv1 = bias_variable('conv1', 32)\n",
    "    x_image = tf.reshape(x, [-1, 28, 28, 1])\n",
    "    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#풀링층1\n",
    "with tf.name_scope('pool1')as scope:\n",
    "    h_pool1 = max_pool(h_conv1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "#합성곱층2\n",
    "with tf.name_scope('conv2')as scope:\n",
    "    W_conv2 = weight_variable('conv2', [5,5,32,64])\n",
    "    b_conv2 = bias_variable('conv2', 64)\n",
    "    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#풀링층2\n",
    "with tf.name_scope('pool2')as scope:\n",
    "    h_pool2 = max_pool(h_conv2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "#전결합층\n",
    "with tf.name_scope('fully_connected')as scope:\n",
    "    n = 7*7*64\n",
    "    W_fc = weight_variable('fc', [n, 1024])\n",
    "    b_fc = bias_variable('fc', 1024)\n",
    "    h_pool2_flat = tf.reshape(h_pool2, [-1, n])\n",
    "    h_fc = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc)+b_fc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-16-5994a86672d9>:4: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n"
     ]
    }
   ],
   "source": [
    "#드롭아웃(오버피팅) 막기\n",
    "with tf.name_scope('dropout')as scope:\n",
    "    keep_prob = tf.placeholder(tf.float32)\n",
    "    h_fc_drop = tf.nn.dropout(h_fc, keep_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#출력층\n",
    "with tf.name_scope('readout')as scope:\n",
    "    W_fc2 = weight_variable('fc2', [1024,10])\n",
    "    b_fc2 = bias_variable('fc2', 10)\n",
    "    y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2)+b_fc2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#모델 학습시키기\n",
    "with tf.name_scope('loss')as scope:\n",
    "    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))\n",
    "with tf.name_scope('training')as scope:\n",
    "    optimizer = tf.train.AdamOptimizer(1e-4)\n",
    "    train_step = optimizer.minimize(cross_entropy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#모델 평가하기\n",
    "with tf.name_scope('predict')as scope:\n",
    "    predict_step = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))\n",
    "    accuracy_step = tf.reduce_mean(tf.cast(predict_step, tf.float32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "#feed_dict 설정하기\n",
    "def set_feed(images, labels, prob):\n",
    "    return {x: images, y_: labels, keep_prob: prob}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "step= 0 loss= 528.302 acc= 0.0649\n",
      "step= 100 loss= 49.35867 acc= 0.8441\n",
      "step= 200 loss= 22.708101 acc= 0.9039\n",
      "step= 300 loss= 10.109159 acc= 0.9287\n",
      "step= 400 loss= 24.118204 acc= 0.9374\n",
      "step= 500 loss= 9.911857 acc= 0.9461\n",
      "step= 600 loss= 19.468376 acc= 0.9518\n",
      "step= 700 loss= 23.404648 acc= 0.9563\n",
      "step= 800 loss= 12.622833 acc= 0.9565\n",
      "step= 900 loss= 4.968046 acc= 0.9627\n",
      "step= 1000 loss= 14.638218 acc= 0.9652\n",
      "step= 1100 loss= 8.284477 acc= 0.9661\n",
      "step= 1200 loss= 13.454403 acc= 0.9683\n",
      "step= 1300 loss= 14.4372425 acc= 0.9712\n",
      "step= 1400 loss= 4.8534 acc= 0.9691\n",
      "step= 1500 loss= 6.1591644 acc= 0.9728\n",
      "step= 1600 loss= 3.0944781 acc= 0.9714\n",
      "step= 1700 loss= 9.677289 acc= 0.9732\n",
      "step= 1800 loss= 3.2155888 acc= 0.9746\n",
      "step= 1900 loss= 2.2452202 acc= 0.9749\n",
      "step= 2000 loss= 9.033638 acc= 0.9762\n",
      "step= 2100 loss= 2.871972 acc= 0.9787\n",
      "step= 2200 loss= 4.46368 acc= 0.9767\n",
      "step= 2300 loss= 10.45441 acc= 0.9788\n",
      "step= 2400 loss= 9.1558 acc= 0.9801\n",
      "step= 2500 loss= 0.9884559 acc= 0.9803\n",
      "step= 2600 loss= 5.075786 acc= 0.9811\n",
      "step= 2700 loss= 9.614168 acc= 0.98\n",
      "step= 2800 loss= 6.50497 acc= 0.9818\n",
      "step= 2900 loss= 1.0217075 acc= 0.982\n",
      "step= 3000 loss= 7.325948 acc= 0.9822\n",
      "step= 3100 loss= 6.980609 acc= 0.9824\n",
      "step= 3200 loss= 2.4198554 acc= 0.9841\n",
      "step= 3300 loss= 5.490091 acc= 0.9829\n",
      "step= 3400 loss= 3.8550763 acc= 0.9823\n",
      "step= 3500 loss= 2.2016366 acc= 0.9823\n",
      "step= 3600 loss= 7.8682117 acc= 0.9834\n",
      "step= 3700 loss= 1.4357958 acc= 0.9833\n",
      "step= 3800 loss= 2.548103 acc= 0.9839\n",
      "step= 3900 loss= 3.8240728 acc= 0.9841\n",
      "step= 4000 loss= 2.161399 acc= 0.9836\n",
      "step= 4100 loss= 2.5896173 acc= 0.9845\n",
      "step= 4200 loss= 0.6509056 acc= 0.9854\n",
      "step= 4300 loss= 1.41986 acc= 0.9845\n",
      "step= 4400 loss= 1.1109413 acc= 0.9859\n",
      "step= 4500 loss= 15.27157 acc= 0.986\n",
      "step= 4600 loss= 0.840525 acc= 0.9862\n",
      "step= 4700 loss= 1.784272 acc= 0.9866\n",
      "step= 4800 loss= 1.1331197 acc= 0.9856\n",
      "step= 4900 loss= 7.8049583 acc= 0.9856\n",
      "step= 5000 loss= 2.8679729 acc= 0.9872\n",
      "step= 5100 loss= 2.0345058 acc= 0.9879\n",
      "step= 5200 loss= 0.9109311 acc= 0.9872\n",
      "step= 5300 loss= 6.25639 acc= 0.9874\n",
      "step= 5400 loss= 4.5127783 acc= 0.9872\n",
      "step= 5500 loss= 0.24910937 acc= 0.987\n",
      "step= 5600 loss= 0.15423322 acc= 0.9871\n",
      "step= 5700 loss= 4.81521 acc= 0.9886\n",
      "step= 5800 loss= 3.8856058 acc= 0.9891\n",
      "step= 5900 loss= 5.2660484 acc= 0.9879\n",
      "step= 6000 loss= 2.6522248 acc= 0.9879\n",
      "step= 6100 loss= 0.24955006 acc= 0.9888\n",
      "step= 6200 loss= 2.1365242 acc= 0.9876\n",
      "step= 6300 loss= 1.9520317 acc= 0.9879\n",
      "step= 6400 loss= 1.3452786 acc= 0.9884\n",
      "step= 6500 loss= 0.38077676 acc= 0.9892\n",
      "step= 6600 loss= 0.5316976 acc= 0.9878\n",
      "step= 6700 loss= 0.70916224 acc= 0.9902\n",
      "step= 6800 loss= 0.98218644 acc= 0.9887\n",
      "step= 6900 loss= 6.000703 acc= 0.9896\n",
      "step= 7000 loss= 1.4223719 acc= 0.9893\n",
      "step= 7100 loss= 0.97525156 acc= 0.9875\n",
      "step= 7200 loss= 0.28736356 acc= 0.9885\n",
      "step= 7300 loss= 0.498018 acc= 0.9896\n",
      "step= 7400 loss= 5.207006 acc= 0.9895\n",
      "step= 7500 loss= 1.9468517 acc= 0.9897\n",
      "step= 7600 loss= 2.83664 acc= 0.9901\n",
      "step= 7700 loss= 1.4523153 acc= 0.9885\n",
      "step= 7800 loss= 0.47903335 acc= 0.99\n",
      "step= 7900 loss= 1.028475 acc= 0.9901\n",
      "step= 8000 loss= 1.450383 acc= 0.9903\n",
      "step= 8100 loss= 1.0609964 acc= 0.9904\n",
      "step= 8200 loss= 4.440122 acc= 0.9906\n",
      "step= 8300 loss= 2.3708305 acc= 0.9893\n",
      "step= 8400 loss= 0.6934662 acc= 0.9901\n",
      "step= 8500 loss= 0.5390171 acc= 0.9902\n",
      "step= 8600 loss= 0.45022625 acc= 0.9906\n",
      "step= 8700 loss= 2.389267 acc= 0.9901\n",
      "step= 8800 loss= 0.3322739 acc= 0.9899\n",
      "step= 8900 loss= 0.56143135 acc= 0.9899\n",
      "step= 9000 loss= 1.4461226 acc= 0.9896\n",
      "step= 9100 loss= 1.4643446 acc= 0.99\n",
      "step= 9200 loss= 0.24529453 acc= 0.9901\n",
      "step= 9300 loss= 0.23457289 acc= 0.9909\n",
      "step= 9400 loss= 0.71308905 acc= 0.99\n",
      "step= 9500 loss= 0.49615633 acc= 0.9909\n",
      "step= 9600 loss= 0.23815805 acc= 0.9893\n",
      "step= 9700 loss= 2.9357505 acc= 0.9893\n",
      "step= 9800 loss= 0.7997612 acc= 0.9904\n",
      "step= 9900 loss= 1.0603833 acc= 0.9905\n"
     ]
    }
   ],
   "source": [
    "#세션 시작하기\n",
    "with tf.Session() as sess:\n",
    "    sess.run(tf.global_variables_initializer())\n",
    "    #TensorBoard 준비하기\n",
    "    tw = tf.summary.FileWriter('log_dir', graph=sess.graph)\n",
    "    #테스트 전용 피드 만들기\n",
    "    test_fd = set_feed(mnist.test.images, mnist.test.labels, 1)\n",
    "    #학습 시작하기\n",
    "    for step in range(10000):\n",
    "        batch = mnist.train.next_batch(50)\n",
    "        fd = set_feed(batch[0], batch[1], 0.5)\n",
    "        _, loss = sess.run([train_step, cross_entropy], feed_dict=fd)\n",
    "        if step % 100 == 0:\n",
    "            acc = sess.run(accuracy_step, feed_dict=test_fd)\n",
    "            print(\"step=\", step, \"loss=\", loss, \"acc=\", acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "oldHeight": 341.86,
   "position": {
    "height": "40px",
    "left": "715px",
    "right": "20px",
    "top": "90px",
    "width": "355px"
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "varInspector_section_display": "none",
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
