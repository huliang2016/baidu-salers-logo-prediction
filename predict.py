# -*- coding: UTF-8 -*-
'''
Created on 2018/5/2
@author: ByRookie
'''
import os
import pickle

import mxnet as mx
import mxnet.ndarray as nd
from mxnet.image import image

from dataloader.augmentation import transform_predict
from model.model import Model

if __name__ == '__main__':
	test_path = './datasets/test/'

	test_file = './datasets/test.txt'
	ctx = mx.gpu(1)
	# pretrianed_model_name = 'resnet50_v2'
	pretrianed_model_name = 'densenet161'
	pretrained = True
	result_file = './pred/%s_05_17.csv' % pretrianed_model_name
	best_model_weight_path = './weights/%s_dense_512_100_crop_331_gray_0.3_colorJitterAug_0.3_dataset_mean_std.params' % pretrianed_model_name
	# best_model_weight_path = './weights/resnet152_v2_basic_augmentation_dataset_mean_std_no_brightness_pca_0.05_0.9555.params'
	# best_model_weight_path = './weights/resnet152_v2_dense_512_100_crop_gray_0.3_brightness_contrast_saturation_0.125_pca_noise_0.1_inter_method_10.params'

	net = Model(pretrained_model_name=pretrianed_model_name, pretrained=pretrained, ctx=ctx)
	net.hybridize()
	net.collect_params().load(best_model_weight_path)

	sorted_ids = list(range(1, 101))
	sorted_ids.sort(key=lambda x: str(x))
	# sorted_ids.remove(13)

	results = {}
	with open(test_file, 'r') as file:
		contents = file.readlines()
		for index, content in enumerate(contents):
			print('%d/%d' % (index, len(contents)), end='\r')
			content = content.replace('\n', '')
			image_path = os.path.join(test_path, content)

			with open(image_path, 'rb') as f:
				img = image.imdecode(f.read())
				data = transform_predict(img)
				out = net(data.as_in_context(ctx))
				out = nd.SoftmaxActivation(out).mean(axis=0)
				results[image_path] = out.asnumpy()
			# pred_class = np.argmax(out.asnumpy())
			# results[content] = out.asnumpy()
			# results.append('%s %d\n' % (content, sorted_ids[pred_class]))
	pickle.dump(results, open('./datasets/%s_pred_test.pickle' % pretrianed_model_name, 'wb'))
# with open(result_file, 'w') as file:
# 	for content in results:
# 		file.write(content)
