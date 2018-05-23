# -*- coding: UTF-8 -*-
'''
Created on 2018/5/1
@author: ByRookie
'''

import random

import cv2
import mxnet.ndarray as nd
from mxnet import image


def resize_longer(img, resize, target_size):
	H, W, _ = img.shape
	_, needH, needW = target_size
	scale = resize / max(H, W)
	newImage = cv2.resize(img, (int(W * scale), int(H * scale)))
	H, W, _ = newImage.shape
	padTop, padLeft = int((needH - H) / 2), int((needW - W) / 2)
	padBot, padRight = needH - padTop - H, needW - padLeft - W
	newImage = cv2.copyMakeBorder(newImage, padTop, padBot, padLeft, padRight, cv2.BORDER_CONSTANT, value=[0, 0, 0])
	return newImage


def rotate_image(img):
	rows, cols, _ = img.shape
	M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(-30, 30), 1)
	return cv2.warpAffine(img, M, (cols, rows))


def transform_train(data, label, data_shape=(3, 363, 363), resize=363):
	# data [height, width, channel]
	# im = data.astype('float32') / 255,
	ctx = data.context
	im = resize_longer(data.asnumpy(), resize, data_shape)
	im = rotate_image(im)
	im = nd.array(im, ctx=ctx)
	im = im.astype('float32') / 255

	auglist = [
		image.RandomCropAug((331, 331)),
		image.ColorJitterAug(0.3, 0.3, 0.3),
		image.RandomGrayAug(0.5),
		# image.ColorNormalizeAug(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		image.ColorNormalizeAug(mean=(0.417, 0.402, 0.366), std=(0.081, 0.079, 0.080)),
	]

	for aug in auglist:
		im = aug(im)
	# im = im.clip(0, 1)
	im = nd.transpose(im, (2, 0, 1))
	return im, nd.array([label], ctx=ctx).asscalar()


# 验证集图片增广，没有随机裁剪和翻转
def transform_val(data, label, data_shape=(3, 331, 331), resize=331):
	# im = data.astype('float32') / 255,
	ctx = data.context
	im = resize_longer(data.asnumpy(), resize, data_shape)
	im = nd.array(im, ctx=ctx)

	im = im.astype('float32') / 255
	auglist = [
		# image.RandomGrayAug(1.0),
		# image.ColorNormalizeAug(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		image.ColorNormalizeAug(mean=(0.417, 0.402, 0.366), std=(0.081, 0.079, 0.080)),
	]

	for aug in auglist:
		im = aug(im)
	# im = im.clip(0, 1)
	im = nd.transpose(im, (2, 0, 1))
	return im, nd.array([label], ctx=ctx).asscalar()


def ten_crop(img, size):
	H, W = size
	iH, iW = img.shape[1:3]

	if iH < H or iW < W:
		raise ValueError('image size is smaller than crop size')

	img_flip = img[:, :, ::-1]
	crops = nd.stack(
		img[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
		img[:, 0:H, 0:W],
		img[:, iH - H:iH, 0:W],
		img[:, 0:H, iW - W:iW],
		img[:, iH - H:iH, iW - W:iW],

		img_flip[:, (iH - H) // 2:(iH + H) // 2, (iW - W) // 2:(iW + W) // 2],
		img_flip[:, 0:H, 0:W],
		img_flip[:, iH - H:iH, 0:W],
		img_flip[:, 0:H, iW - W:iW],
		img_flip[:, iH - H:iH, iW - W:iW],
	)
	return crops


# 预测集
def transform_predict(data, data_shape=(3, 331, 331), resize=331):
	ctx = data.context
	im = resize_longer(data.asnumpy(), resize, data_shape)
	im = nd.array(im, ctx=ctx)

	im = im.astype('float32') / 255
	auglist = [
		# image.RandomGrayAug(1.0),
		image.ColorNormalizeAug(mean=(0.417, 0.402, 0.366), std=(0.081, 0.079, 0.080)),
	]

	for aug in auglist:
		im = aug(im)
	im = nd.transpose(im, (2, 0, 1))
	return nd.stack(im)
