# -*-coding:utf-8 -*-
''''''
# 项目分析总结
    # 项目学习目标：根据车载摄像头的画面，自动判断如何打方向盘（使用端到端（end-to-end）的深度学习模式）
        #端到端模式补充：端到端指的是输入的是原始数据，输出的是最后的结果。以前有一些数据处理系统或者学习系统输入端不是直接的原始数据，而是在原始数据中提取的特征，
            #这一点在图像问题上尤为突出，因为图像像素数太多，数据维度高，会产生维度灾难，所以原来一个思路是手工提取图像的一些关键特征，这实际就是就一个降维的过程。
            #那么问题来了，特征怎么提？特征提取的好坏异常关键，甚至比学习算法还重要，举个例子，对一系列人图片的数据分类，分类结果是性别，如果你提取的特征是头发的颜色，无论分类算法如何，分类效果都不会好，如果你提取的特征是头发的长短，这个特征就会好很多，但是还是会有错误，如果你提取了一个超强特征，比如染色体的数据，那你的分类基本就不会错了。
            #那么端到端深度学习就是忽略所有这些不同的阶段，用单个神经网络代替它。以语音识别为例，你的目标是输入x，比如说一段音频，然后把它映射到一个输出y，就是这段音频的听写文本。所以和这种有很多阶段的流水线相比，端到端深度学习做的是，你训练一个巨大的神经网络，输入就是一段音频，输出直接是听写文本。
    # 项目数据分析：
        #1、读取和显示数据
        #2、分析数据(面临的问题)
            #数据层面
                #(1)数据不平衡：数据不平衡容易导致学习出来的模型偏向某个答案，而失去预测的意义
                    #方向盘角度绝大部分是0
                    #相对于负的角度，角度为正的情况比较多
                #(2)感兴趣区域：图像中有些区域是多余的，比如天空，如果我们剪切掉这部分，可以减少数据计算量，同时模型训练不会被这种多余的像噪音一样的区域影响，能够集中于关键区域训练
                #(3)图片亮度问题:在真实情景下，开车可能是白天，可能是晚上，亮度不同，如果都是用相同亮度的图片训练，那么模型的鲁棒性很差
                #(4)数据中有左中右三个角度的摄像头拍摄的数据
                #(5)图像是否需要正规化
            #模型层面
                #(1)本项目是一个回归问题，预测连续值——方向盘角度
                #(2)网络模型构建
                    #应该设计多少层，每一层都是什么具体层
                    #如果是卷积层，卷积核的大小，步长大小，激活函数是什么
                    #如果是全连接层，需不需要加正则化
                    #使用什么样的优化器（optimizer） ，优化器的学习率是如何设定的
                #(3)过拟合与欠拟合：如果是过拟合如何调整网络结构，如果欠拟合如何调整网络结构
                #(4)模型评估：使用哪种评估方式和损失函数已经根据模型评估相应调整模型
        #3、问题的解决：
            #数据层面：
                #(1)方向盘角度绝大部分是0:随机去除角度为0的图片数据
                #(2)角度为正多：图形水平翻转，使得正负角度数据的数量相同
                #(3)多余区域：图像切割（感兴趣的区域）
                    #根据经验得到从底向上切20像素，将车头部分的图像切掉
                    #根据经验得到从上向下切80像素，将远方地平线以上的区域切掉
                    #切割后图像为(80, 260)像素
                #(4)图片亮度：图片亮度调整
                    #将图像从RGB色彩空间转换为HSV色彩空间：图像的颜色空间有很多种，在计算机中存储图像一般使用RGB,   但是RGB不适合人的肉眼感知的色彩空间，V代表了亮度
                    #保持HS的值不变，将V的值乘以一个系数[0.1, 1]:将V的值乘以一个【0.1,1】之间的系数，这样可以取到不同亮度的图
                    #将HSV图像转化为RGB
                #(5)三个摄像头角度问题：找到三个摄像头角度的关系
                    #左：中间—0.25   右：中间+0.25
                #(6)图像是正规化
                    #X = X/225 - 0.5
                    #图像的均值从127.5变成0， 范围从[0, 225]变成[-0.5,0.5]
            #模型层面：
                #(1)回归问题:选择回归问题的评估方式和损失函数
                #(2)网络模型构建：借鉴别人的网咯
                #(3)过拟合与欠拟合 ：看训练数据集合和测试数据集loss的走势，是过拟合还是欠拟合，在进行相应的更改
                #（4）模型评估
import csv
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
import cv2
import math
from keras import callbacks
import pickle
import matplotlib.pyplot as plt
import keras.backend as K
from tensorflow_tanxinkeji_works.Preject3_方向盘转动角度预测.YXM_effnet import Effnet
SEED = 13


def r_square(y_true, y_pred):
    SSR = K.mean(K.square(y_pred-K.mean(y_true)),axis=0)
    SST = K.mean(K.square(y_true-K.mean(y_true)),axis=0)
    return SSR/SST


def show_random_samples(X_train, y_train, n):
    '''
    分类任务（图片）：随机展示每种样本图片
    :param X_train: 训练数据
    :param y_train: 标签数据
    :param n_classes: 展示几个样本，可以调节(注意：展示样本个数必须小于y_train[x])
    '''
    rows, cols = 4, 12                    #  一共有4*12个子图, 可以调节
    fig, ax_array = plt.subplots(rows, cols)    #fig：画板 ；ax_array：子图和集合，用二维矩阵表示
    plt.suptitle('Random Samples (one per class)')
    for class_idx, ax in enumerate(ax_array.ravel()):
        if class_idx < n:
            # show a random image of the current class
            cur_X = X_train[y_train == class_idx]
            cur_img = cur_X[np.random.randint(len(cur_X))]
            ax.imshow(cur_img)                       #X : array_like, shape (n, m) or (n, m, 3) or (n, m, 4)
            ax.set_title('{:02d}'.format(class_idx))
        else:
            ax.axis('off')
    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)  #设置
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()


def get_model(shape, effnet = False):
    if effnet == False:
        model = Sequential()

        model.add(Conv2D(8, (5, 5), padding='valid', strides=(1, 1), activation='relu', input_shape=shape))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(8, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(16, (4, 4), padding='valid', strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Conv2D(16, (5, 5), padding='valid', strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(2, 2))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, activation='linear'))  #最后全连接层输出1，只有一个预测值，激活函数为linear,因为是线性的

        # sgd = SGD(lr=0.000001)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=[r_square])   #loss='mean_squared_error'：回归问题的损失函数
        return model
    else:
        model = Effnet(shape, regression_tesk= True)

        return model

def horizontal_flip(img, degree):
    '''
    按照50%的概率水平翻转图像
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # 调整亮度v:alpha * v
    alpha = np.random.uniform(low=0.1, high=1.0, size=None)
    v = img[:, :, 0]
    v = v * alpha
    img[:, :, 0] = v.astype('uint8')   #补充：v = v *alph之后，变成浮点型，必须转化为unit8
    img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_YUV2BGR )

    return img, degree


def random_brightness(img, degree):
    '''
    随机调整输入图像的亮度， 调整强度于 0.1(变黑)和1(无变化)之间
    img: 输入图像
    degree: 输入图像对于的转动角度
    '''
    choice = np.random.choice([1, 0])
    if choice == 1:
        img, degree = cv2.flip(img, 1), -degree
    return img, degree


def left_right_random_swap(img_address, degree, degree_corr=1.0 / 4):
    '''
    随机从左， 中， 右图像中选择一张图像， 并相应调整转动的角度
    img_address: 中间图像的的文件路径
    degree: 中间图像对于的方向盘转动角度
    degree_corr: 方向盘转动角度调整的值
    '''
    swap = np.random.choice(['L', 'R', 'C'])
    if swap == 'L':
        img_address = img_address.replace('center', 'left')
        degree = np.arctan(math.tan(degree) + degree_corr)
        return img_address, degree
    elif swap == 'R':
        img_address = img_address.replace('center', 'right')
        degree = np.arctan(math.tan(degree) + degree_corr)
        return img_address, degree
    else:
        return img_address, degree


def discard_zero_steering(degrees, rate):
    '''
    从角度为零的index中随机选择部分index返回
    degrees: 输入的角度值
    rate: 丢弃率， 如果rate=0.8， 意味着80%的index会被返回， 用于丢弃
    '''
    degrees_del_index = np.where(degrees == 0)
    degrees_del_index = degrees_del_index[0]
    degrees_del_size = int(len(degrees_del_index) * rate)

    return np.random.choice(degrees_del_index, size=degrees_del_size, replace=False)


def image_transformation(img_address, degree, data_dir):
    img_address, degree = left_right_random_swap(img_address, degree)
    img = cv2.imread(data_dir + img_address)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, degree = random_brightness(img, degree)
    img, degree = horizontal_flip(img, degree)

    return img, degree


def batch_generator(x, y, batch_size, shape, training=True, data_dir='data/', monitor=True, yieldXY=True,
                    discard_rate=0.95):
    """
    产生批处理的数据的generator
    x: 文件路径list
    y: 方向盘的角度
    training: 值为True时产生训练数据
              值为True时产生validation数据
    batch_size: 批处理大小
    shape: 输入图像的尺寸(高, 宽, 通道)
    data_dir: 数据目录, 包含一个IMG文件夹
    monitor: 保存一个batch的样本为 'X_batch_sample.npy‘ 和'y_bag.npy’
    yieldXY: 为True时, 返回(X, Y)
             为False时, 只返回 X only
    discard_rate: 随机丢弃角度为零的训练数据的概率
    """

    if training:
        y_bag = []
        x, y = shuffle(x, y)
        rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
        new_x = np.delete(x, rand_zero_idx, axis=0)
        new_y = np.delete(y, rand_zero_idx, axis=0)
    else:
        new_x = x
        new_y = y

    offset = 0
    while True:
        X = np.empty((batch_size, *shape))
        Y = np.empty((batch_size, 1))

        for example in range(batch_size):
            img_address, img_steering = new_x[example + offset], new_y[example + offset]

            if training:
                img, img_steering = image_transformation(img_address, img_steering, data_dir)
            else:
                img = cv2.imread(data_dir + img_address)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            X[example, :, :, :] = cv2.resize(img[80:140, 0:320], (shape[0], shape[1])) / 255 - 0.5   # #图像切割之后，大小改变，必须缩放回网络模型能够接受的大小。

            Y[example] = img_steering
            if training:
                y_bag.append(img_steering)

            '''
             到达原来数据的结尾, 从头开始
            '''
            if (example + 1) + offset > len(new_y) - 1:
                x, y = shuffle(x, y)
                rand_zero_idx = discard_zero_steering(y, rate=discard_rate)
                new_x = x
                new_y = y
                new_x = np.delete(new_x, rand_zero_idx, axis=0)
                new_y = np.delete(new_y, rand_zero_idx, axis=0)
                offset = 0
        if yieldXY:
            yield (X, Y)
        else:
            yield X

        offset = offset + batch_size
        if training:
            np.save('y_bag.npy', np.array(y_bag))
            np.save('Xbatch_sample.npy', X)



if __name__ == '__main__':
    data_path = 'data/'
    with open(data_path + 'driving_log.csv', 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        log = []
        for row in file_reader:
            log.append(row)

    log = np.array(log)
    # 去掉文件第一行
    log = log[1:, :]

    # 判断图像文件数量是否等于csv日志文件中记录的数量
    ls_imgs = glob.glob(data_path + 'IMG/*.jpg')
    assert len(ls_imgs) == len(log) * 3, 'number of images does not match'

    # 使用20%的数据作为测试数据
    validation_ratio = 0.2
    shape = (128, 128, 3)
    batch_size = 32
    nb_epoch = 40

    x_ = log[:, 0]
    y_ = log[:, 3].astype(float)
    x_, y_ = shuffle(x_, y_)

    X_train, X_val, y_train, y_val = train_test_split(x_, y_, test_size=validation_ratio, random_state=SEED)

    print('batch size: {}'.format(batch_size))
    print('Train set size: {} | Validation set size: {}'.format(len(X_train), len(X_val)))

    # samples_per_epoch = batch_size
    samples_per_epoch = 1000
    # 使得validation数据量大小为batch_size的整数陪
    nb_val_samples = len(y_val) - len(y_val) % batch_size
    model = get_model(shape, effnet = True)
    print(model.summary())

    # 根据validation loss保存最优模型
    save_best = callbacks.ModelCheckpoint(filepath= 'train_modelSave\effnet_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                          monitor='val_loss', verbose=1,
                                          save_best_only=True, save_weights_only=False,
                                          mode='min', period=1, )
    # 调整学习率
    learnRate = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto',
                                      epsilon=0.0001, cooldown=0, min_lr=0)
    # 如果训练持续没有validation loss的提升, 提前结束训练
    early_stop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                         verbose=0, mode='auto')
    callbacks_list = [early_stop, save_best, learnRate]

    history = model.fit_generator(batch_generator(X_train, y_train, batch_size, shape, training=True),
                                  steps_per_epoch=samples_per_epoch,
                                  validation_steps=nb_val_samples // batch_size,
                                  validation_data=batch_generator(X_val, y_val, batch_size, shape,
                                                                  training=False, monitor=False),
                                  epochs=nb_epoch, verbose=1, callbacks=callbacks_list)

    # with open('./trainHistoryDict.p', 'wb') as file_pi:
    #     pickle.dump(history.history, file_pi)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig('train_val_loss.jpg')
    plt.show()
    # 保存模型
    # with open('model.json', 'w') as f:
    #     f.write(model.to_json())
    # model.save('model.h5')
    print('_____Train Done!______')




#补充：
# (1)
#numpy.random.choice(a, size=None, replace=True, p=None):功能：随机在提供的数组a中, 选择size元素个数返回
    #从a(只要是ndarray都可以，但必须是一维的,或者一个整形数字)中随机抽取数字，并组成指定大小(size)的数组
    #replace:True表示可以取相同数字，False表示不可以取相同数字
    #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
    #返回一个列表
# (2)
#cv2.flip语法：flip(src, flipCode[, dst])
    # 1： 水平翻转   0：垂直翻转       -1： 水平垂直翻转
# (3)
# 函数原型：  numpy.random.uniform(low, high, size)
#     功能：从一个均匀分布[low, high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
#     参数介绍: 
#         low: 采样下界，float类型，默认值为0；
#         high: 采样上界，float类型，默认值为1；
#         size: 输出样本形状，为int或元组(tuple)类型，例如，size = (m, n, k), 则输出m * n * k个样本，缺省时输出1个值。
#(4)

# (5)
# Python replace()方法，描述：
    # Python replace() 方法把字符串中的 old（旧字符串） 替换成 new(新字符串)，如果指定第三个参数max，则替换不超过 max 次。
    # 语法：replace()方法语法：str.replace(old, new[, max])
    # 参数：
    # old - - 将被替换的子字符串。
    # new - - 新字符串，用于替换old子字符串。
    # max - - 可选字符串, 替换不超过max次
    # 返回值：返回字符串中的 old（旧字符串）替换成new(新字符串)后生成的新字符串，如果指定第三个参数max，则替换不超过max次。
# (6)
# numpy.where() 用法详解：numpy.where (condition[, x, y])
     #numpy.where() 有两种用法：
        # (1)
            # np.where(condition, x, y)
            # 满足条件(condition)，输出x，不满足输出y。
        # (2)
            #np.where(condition)
            #只有条件(condition)，没有x和y，则输出满足条件(即非0)
            #元素的坐标(等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
            #返回一个带有索引的元组
            # 例子：
            # arr = np.array([0, 2, 2, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0]) 注意arr一定是ndarray结构
            # print(np.where(arr == 0))
            # 例子：#where的用法之一
                # a = np.arange(10)
                # a = np.where(a < 5)
                # print(a)
                # print(type(a))
                # a = a[0]  #将元组转化为ndarray
                # print(a)
                # print(type(a))
                # b = np.arange(20).reshape(4, 5)  #区别与a中的行向量，where会返回正确的索引；但是在数组中不会，返回的都是0
                # b = np.where(b < 5)
                # print(b)
                # print(b[0])
# (6)Opencv中图像的缩放 resize（）函数的应用：https://www.cnblogs.com/jyxbk/p/7651241.html
    # cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
    # 参数说明：
        #src - 原图
        #dst - 目标图像。当参数dsize不为0时，dst的大小为size；否则，它的大小需要根据src的大小，参数fx和fy决定。dst的类型（type）和src图像相同
        #dsize - 目标图像大小。当dsize为0时，它可以通过公式计算得出
            # 参数dsize和参数(fx, fy)不能够同时为0， fx - 水平轴上的比例因子。fy - 垂直轴上的比例因子。
        #interpolation - 插值方法。共有5种：
            #１）INTER_NEAREST - 最近邻插值法
            #２）INTER_LINEAR - 双线性插值法（默认）
            #３）INTER_AREA - 基于局部像素的重采样（resampling  using pixel area relation）。对于图像抽取（image decimation）来说，这可能是一个更好的方法。但如果是放大图像时，它和最近邻法的效果类似。
            #４）INTER_CUBIC - 基于4x4像素邻域的3次插值法
            #５）INTER_LANCZOS4 - 基于8x8像素邻域的Lanczos插值

# 按照比例缩放：PIL库中Image类thumbnail方法
    #thumbnail(size, resample)(创建缩略图)
    #im.thumbnail((50, 50), resample=Image.BICUBIC)
    #im.show()
    #上面的代码可以创建一个指定大小(size)的缩略图。
    #需要注意的是，thumbnail方法是原地操作，返回值是None。第一个参数是指定的缩略图的大小，第二个是采样的，有Image.BICUBIC，PIL.Image.LANCZOS，PIL.Image.BILINEAR，PIL.Image.NEAREST这四种采样方法。默认是Image.BICUBIC。

# PIL库中Image类thumbnail方法和resize方法的比较
    # resize()方法可以缩小也可以放大，而thumbnail()方法只能缩小；
    # resize()方法不会改变对象的大小，只会返回一个新的Image对象，而thumbnail()方法会直接改变对象的大小，返回值为none；
    # resize()方法中的size参数直接规定了修改后的大小，而thumbnail()方法按比例缩小，size参数只规定修改后size的最大值。

# np.resize和np.reshape的区别
    #二者都是改变输入的形状，但是区别是： reshape只能改变形状，不能改变原始输入包含的元素个数。resize可以改变尺寸。
    #resize在尺寸缩小时，只保留一部分元素；在尺寸扩大的时候，它会重复使用原始输入的元素值进行填充（也可以通过a.resize()的方式实现用0进行填充）。
    #特别注意的一点是：这两个函数只是改变形状，虽然resize可以改变尺寸，但是只是进行简单的裁剪和填充。如果想更加精确的使用插值的方式进行尺寸的扩大或缩小，比如改变一副图像的形状，则不能用这个函数进行。
# (7)
# (8)glob.glob()
    # 是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件，就类似于Windows下的文件搜索，支持通配符操作，,?,[]这三个通配符，代表0个或多个字符，?代表一个字符，[]匹配指定范围内的字符，如[0-9]匹配数字。
    # 它的主要方法就是glob,该方法返回所有匹配的文件路径列表，该方法需要一个参数用来指定匹配的路径字符串（本字符串可以为绝对路径也可以为相对路径），其返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件。
    # 例子：glob.glob(r'c:\*.txt')：获得C盘下的所有txt文件
# (9)Python实现图片裁剪的两种方式——Pillow和OpenCV
    #OpenCV对其进行裁剪
        #我们先用imread方法读取待裁剪的图片，然后查看它的shape，shape的输出是(1080, 1920,3)，输出的顺序的是高度、宽度、通道数。之后我们利用数组切片的方式获取需要裁剪的图片范围。
        # 这里需要注意的是切片给出的坐标为需要裁剪的图片在原图片上的坐标，顺序为[y0:y1, x0:x1]，其中原图的左上角是坐标原点。最后我们用cv2.imwrite()方法将裁剪得到的图片保存到本地（第一个参数为图片名，第二参数为需要保存的图片）
        # import cv2
        # img = cv2.imread("./data/cut/thor.jpg")
        # print(img.shape)
        # cropped = img[0:128, 0:512]  # 裁剪坐标为[y0:y1, x0:x1]
        # cv2.imwrite("./data/cut/cv_cut_thor.jpg", cropped)
    #Pillow对其进行裁剪
        # 首先我们使用open方法读取图片，然后查看它的size（这里的size和OpenCV中的shape是类似的），size的输出是(1920, 1080)，也就是图片的宽度和高度。
        #之后我们调用crop方法来对图片进行裁剪，crop需要给定一个box参数，box是一个四元组，元组中元素的顺序是需要裁剪得到的图片在原图中的左、上、右、下坐标，即(left, upper, right, lower)。
        #然后，我们使用save方法保存裁剪得到的图片
        # from PIL import Image
        # img = Image.open("./data/cut/thor.jpg")
        # print(img.size)
        # cropped = img.crop((0, 0, 512, 128))  # (left, upper, right, lower)
        # cropped.save("./data/cut/pil_cut_thor.jpg")
# (8)图像归一化：https://blog.csdn.net/u010555688/article/details/25551255
               #https: // www.sohu.com / a / 243383119_823210
#(9)Numpy数组的保存与读取：
    # np.save和np.load
    # np.savetxt和np.loadtxt
    # tofile
    # https: // blog.csdn.net / u010089444 / article / details / 52738479
#(10)glob.glob() 函数
    # 用它可以查找符合特定规则的文件路径名。跟使用windows下的文件搜索差不多。查找文件只用到三个匹配符：””, “?”, “[]”。””匹配0个或多个字符；
    # ”?”匹配单个字符；”[]”匹配指定范围内的字符，如：[0 - 9]匹配数字。
    # glob.glob() 函数返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。
    # 下面是使用glob.glob的例子：
    # import glob
    # 获取指定目录下的所有图片
    # print(glob.glob(r"/home/qiaoyunhao/*/*.png"), "\n")  # 加上r让字符串不转义
    # 获取上级目录的所有.py文件
    # print(glob.glob(r'../*.py'))  # 相对路径
#(11)Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
    # 断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况，例如我们的代码只能在 Linux 系统下运行，可以先判断当前系统是否符合条件。
        # 语法格式如下：assert expression
        #     例子：import sys
        #         assert ('linux' in sys.platform), "改代码只能在 Linux 下执行"    # 条件为 true 正常执行     # 条件为 false 触发异常
# (12)yield 和 return的区别
    '''
    yield 生成器：生成一个迭代器
        yield的作用是把一个函数变成一个generator迭代器
        yield可以与return做类比：
            retrun:返回一个值 ,结束函数的运行
            yield:把一个函数变成一个迭代器, 在使用next()调用时这个迭代器时，函数执行将暂停，然后返回一次，再调用时，才从上一次暂停的地方执行，然后再暂停，依次类推
        有两种遍历操作：next和for循环
        使用生成器可以达到延迟操作的效果，所谓延迟操作就是指在需要的时候产生结果而不是立即产生就结果，节省资源消耗，和声明一个序列不同的是生成器，在不使用的时候几乎是不占内存的。
    '''
    # def getNum(n):
    #     i = 0
    #     while i <= n:
    #         #return i     #返回一个i ,结束函数的运行
    #         yield i     #将函数变成一个generator，在调用这个生成器时，调用一次，返回一次，可以多次返回，不调用时，则不返回，可以节省内存空间
    #         i+=1
    # print(getNum(5))
    # a = getNum(5)  #把生成器赋值给一个变量a


# 思考：
# (1)关于数据的读取
     #在preject2中，导入模块 import pickle， 再使用train = pickle.load(f)读取文件。因为文件是.p文件。
     #在preject3中，导入模块 import csv，再使用file_reader = csv.reader(csvfile, delimiter=',')读取文件，因为文件是csv类型。
#思考2：为什么compile中参数没有metrics=['accuracy']？？？？？？？？？？？？？？？？？？？？？？？？
# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[r2_score_dfy])？？？？？？？？？？？？？？？？？？？？？？
# 回归问题的 度量指标  r2_score
# 加上自定义metrics
