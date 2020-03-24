import argparse
import base64
import json
from keras import backend as K
import numpy as np
import socketio         #给模拟器通讯使用
import eventlet         #给模拟器通讯使用
import eventlet.wsgi    #给模拟器通讯使用
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template    #Flask：网络框架，方便我们写相关的网络程序
from io import BytesIO
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import cv2
# Fix error with Keras and TensorFlow
#import tensorflow as tf
from tensorflow_tanxinkeji_works.Preject3_方向盘转动角度预测.YXM_train_Baseline import r_square
#tf.python.control_flow_ops = tf
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))
sio = socketio.Server()   #启动一个socketio的服务器
app = Flask(__name__)     #创建一个网络框架的APP
model = None              #全局变量：单例模式
prev_image_array = None


# initialize parameter to record steering angle
# idx = 0
# steering_pred = []

@sio.on('telemetry')
def telemetry(sid, data):
    # global idx
    # global steering_pred
    # The current steering angle of the car：当前车的方向盘转动角度
    steering_angle = data["steering_angle"]
    # The current throttle of the car：当前油门的大小
    throttle = data["throttle"]
    # The current speed of the car：车速
    speed = data["speed"]
    # The current image from the center camera of the car：车中间位置的摄像头捕捉的画面
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))  #画面是用字符串的方式编码的，我们需要用base64的方式来进行解码
    image_array = np.asarray(image)
    # convert from BGR to RGB：色彩空间转换：BGR到RGB
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    # save frames locally
    # fname = 'test_img/fname' + str(idx) +'.jpg'
    # cv2.imwrite(fname, image_array)

    # resize image to match training data image size  #缩放图像到网络输入要求的大小
    image_array = image_array[80:140, 0:320]
    image_array = cv2.resize(image_array, (128, 128)) / 255. - 0.5  #正规化图像
    transformed_image_array = image_array[None, :, :, :]    #将图像从3维增加到一个批处理维度

    #
    # print(transformed_image_array.shape)
    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    steering_angle = float(model.predict(transformed_image_array, batch_size=1))  #预测角度
    # update steering angle record
    # steering_pred.append(steering_angle)
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 1  #设置油门为常数0.1

    # idx = idx + 1
    # np.save('test_steering.npy', np.array(steering_pred))
    print(steering_angle, throttle)  #输出预测的角度和油门到命令行
    send_control(steering_angle, throttle)  #发送方向盘转动角度和油门给汽车模拟器


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):  #这个方法接受方向盘转动的角度和油门的大小
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    model = load_model('train_modelSave\epoch-30_loss-0.0297_val_loss-0.0532.h5', custom_objects={'r_square' : r_square})
    # parser = argparse.ArgumentParser(description='Remote Driving')
    # parser.add_argument('model', type=str,
    #                     help='Path to model definition json. Model weights should be on the same path.')
    # args = parser.parse_args()
    # with open(args.model, 'r') as jfile:
    #     # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
    #     # then you will have to call:
    #     #
    #     #   model = model_from_json(json.loads(jfile.read()))\
    #     #
    #     # instead.
    #     model = model_from_json(jfile.read())
    #
    # model.compile("adam", "mse")
    # weights_file = args.model.replace('json', 'h5')
    # model.load_weights(weights_file)
    #
    # wrap Flask application with engineio's middleware：将Flask应用绑定到中间件上
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server：启动eventlet WSGI 服务器， 监听4567端口
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)