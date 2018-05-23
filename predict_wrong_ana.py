# -*- coding: UTF-8 -*-
'''
Created on 2018/5/2
@author: ByRookie
'''
import os
import pickle

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.image import image

from dataloader.augmentation import transform_predict
from model.model import Model

if __name__ == '__main__':
	os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
	val_path = './datasets/valid/'

	ctx = mx.gpu(1)
	pretrianed_model_name1 = 'inceptionv3'
	# pretrianed_model_name1 = 'inceptionV3'
	pretrained = True
	best_model_weight_path1 = './weights/%s_dense_512_100_crop_331_gray_0.3_colorJitterAug_0.3_dataset_mean_std.params' % pretrianed_model_name1
	# best_model_weight_path1 = './weights/inceptionv3_dense_512_100_crop_gray_0.5_brightness_contrast_saturation_0.125.params'

	net1 = Model(out_classes=100, pretrained_model_name=pretrianed_model_name1, pretrained=pretrained, ctx=ctx)
	net1.hybridize()
	net1.collect_params().load(best_model_weight_path1)

	sorted_ids = list(range(1, 101))
	sorted_ids.sort(key=lambda x: str(x))
	# sorted_ids.remove(13)

	results = {}
	right, count = 0, 0
	val_dirs = os.listdir(val_path)
	for index, dirs in enumerate(val_dirs):
		# print('%d/%d' % (index, len(val_dirs)))
		tempClass = dirs
		# if tempClass == '13':
		# 	tempClass = '63'
		dirs = os.path.join(val_path, dirs)
		for image_path in os.listdir(dirs):
			image_path = os.path.join(dirs, image_path)

			with open(image_path, 'rb') as f:
				img = image.imdecode(f.read())
				data = transform_predict(img)
				out1 = net1(data.as_in_context(ctx))
				out1 = nd.SoftmaxActivation(out1).mean(axis=0)
				pred_class = np.argmax(out1.asnumpy())
				results[image_path] = out1.asnumpy()
				count += 1
				# outnp = out1.asnumpy()
				# argsorts = np.argsort(outnp)
				# print(sorted_ids[argsorts[-5]], sorted_ids[argsorts[-4]], sorted_ids[argsorts[-3]],
				#       sorted_ids[argsorts[-2]], sorted_ids[argsorts[-1]], outnp[argsorts[-5:]])
				# print(image_path, sorted_ids[pred_class], tempClass, '\n\n')
				if sorted_ids[pred_class] == int(tempClass):
					right += 1
				else:
					print(image_path, sorted_ids[pred_class], out1.asnumpy()[pred_class], tempClass)

	print('%d/%d, %.4f' % (right, count, float(right) / count))
	pickle.dump(results, open('./datasets/%s_pred.pickle' % pretrianed_model_name1, 'wb'))
