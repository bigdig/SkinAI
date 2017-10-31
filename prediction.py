#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import keras
import numpy as np
from model.skin_resnet import skin_resnet
from keras.preprocessing import image
from keras.preprocessing.image import transform_matrix_offset_center, apply_transform


def img_rotation(x, theta,row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='nearest', cval=0.):
    """modifed from keras random_rotation
    # Arguments
        x: Input tensor. Must be 3D.
        theta: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        Rotated Numpy image tensor.
    """
    # theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def img_norm(img_array):

	"""tensorflow tensor form
	"""
	img_array = img_array.reshape((1,) + img_array.shape)
	# normalization:
	for i in range(img_array.shape[0]):
		for k in range(3):
			img_array[i,::,::,k] -= np.mean(img_array[i,::,::,k])
			img_array[i,::,::,k] /= np.std(img_array[i,::,::,k]) + 1e-7

	return img_array


def pred(image_path):
	sz = 224 # resize image into (224,224,3)
	theta = [0,-90,90,120,270]
	preds = np.ndarray(shape=(5), dtype=float)

	model = skin_resnet()  # 导入网络
	model.load_weights("model_weight/resnet50_turnable_152_d2_3.h5") # 导入权重

	img = image.load_img(image_path, target_size = (sz,sz))
	img_array = image.img_to_array(img)
    
	i = 0
    
	for thet in theta:
		img_array2 = img_rotation(img_array, thet)
		img_array2 = img_norm(img_array2)
		results = model.predict(img_array2) * 100
		preds[i] = results[0,0]
		i = i + 1
	pred_f = np.median(preds) # pred_f是预测结果

	print ("*"*50)
	print "皮肤癌相似度：" + str("%.2f") %pred_f +"%"
	print ("*"*50)

	return pred_f

if __name__ == '__main__':


	print ("请输入图像所在位置:")
	image_path = raw_input()  # 输入图片

	pred(image_path)
