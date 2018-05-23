# -*- coding: UTF-8 -*-
'''
Created on 2018/5/2
@author: ByRookie
'''

import os

import mxnet as mx


def getImages(image_dir):
	images = []
	for (dirpath, dirnames, filenames) in os.walk(image_dir):
		for filename in filenames:
			if filename.endswith('.jpg'):
				images.append(os.path.join(dirpath, filename))
	return images


if __name__ == '__main__':
	base_dir = '../datasets/train/'
	images = getImages(base_dir)

	r = 0  # r mean
	g = 0  # g mean
	b = 0  # b mean

	r_2 = 0  # r^2
	g_2 = 0  # g^2
	b_2 = 0  # b^2

	total = 0
	for index, img_name in enumerate(images):
		if index % 100 == 0:
			print('%d/%d' % (index, len(images)))
		# ndarray, width x height x 3
		img = mx.image.imread(img_name)
		img = img.astype('float32') / 255.
		total += img.shape[0] * img.shape[1]

		r += img[:, :, 0].sum().asscalar()
		g += img[:, :, 1].sum().asscalar()
		b += img[:, :, 2].sum().asscalar()

		r_2 += (img[:, :, 0] ** 2).sum().asscalar()
		g_2 += (img[:, :, 1] ** 2).sum().asscalar()
		b_2 += (img[:, :, 2] ** 2).sum().asscalar()

	r_mean = r / total
	g_mean = g / total
	b_mean = b / total

	r_var = r_2 / total - r_mean ** 2
	g_var = g_2 / total - g_mean ** 2
	b_var = b_2 / total - b_mean ** 2

	# 0.41701717234891394 0.40207801064316784 0.3661300156275214
	# 0.08123444628965928 0.07876823024930732 0.07964408191238109
	print(r_mean, g_mean, b_mean)
	print(r_var, g_var, b_var)
