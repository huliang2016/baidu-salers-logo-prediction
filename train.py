# -*- coding: UTF-8 -*-
'''
Created on 2018/5/1
@author: ByRookie
'''
import logging
import math
import os
import time

import mxnet as mx
import mxnet.autograd as ag
import mxnet.ndarray as nd
import numpy as np
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageFolderDataset

from dataloader.augmentation import transform_train, transform_val
# params
from model.model import Model


def progressbar(i, n, bar_len=40):
	percents = math.ceil(100.0 * i / float(n))
	filled_len = int(round(bar_len * i / float(n)))
	prog_bar = '=' * filled_len + '-' * (bar_len - filled_len)
	print('[%s] %s%s' % (prog_bar, percents, '%'), end='\r')


def validate(net, val_data, ctx):
	metric = mx.metric.Accuracy()
	L = gluon.loss.SoftmaxCrossEntropyLoss()
	val_loss = 0
	for i, batch in enumerate(val_data):
		data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
		label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
		outputs = [net(X) for X in data]
		metric.update(label, outputs)
		loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
		val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
	_, val_acc = metric.get()
	return val_acc, val_loss / len(val_data)


def label_transform(label, classes):
	ind = label.astype('int')
	res = nd.zeros((ind.shape[0], classes), ctx=label.context)
	res[nd.arange(ind.shape[0], ctx=label.context), ind] = 1
	return res


if __name__ == '__main__':
	os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
	os.environ["CUDA_VISIBLE_DEVICES"] = '3'
	train_path = './datasets/train/'
	val_path = './datasets/valid/'

	# hyper params
	pretrianed_model_name = 'inceptionv3'
	pretrained = True
	ctx = [mx.gpu()]
	batch_size = 16
	lr = 1e-3
	momentum = 0.9
	wd = 1e-5
	epochs = 100
	MAX_IMPROVE_COUNT = 10
	MIX_UP = False
	alpha = 1
	classes = 100

	# logging params
	logging.basicConfig(level=logging.INFO,
	                    handlers=[
		                    logging.StreamHandler(),
		                    logging.FileHandler(
			                    './logs/%s_shape_331_dataset_mean_std.log' % pretrianed_model_name)
	                    ])
	logging.info(
		'100 classes, RandomCropAug = 331, rand_gray = 0.3 , ColorJitterAug(0.3, 0.3, 0.3), dataset_mean_std.')
	model_weight_save_path = './weights/%s_dense_512_100_crop_331_gray_0.3_colorJitterAug_0.3_dataset_mean_std.params' % pretrianed_model_name

	# 定义训练集的 DataLoader
	train_data = DataLoader(
		ImageFolderDataset(train_path, transform=transform_train),
		batch_size=batch_size, shuffle=True, num_workers=4)

	# 定义验证集的 DataLoader
	val_data = DataLoader(
		ImageFolderDataset(val_path, transform=transform_val),
		batch_size=batch_size, shuffle=False, num_workers=4)

	# 定义网络
	net = Model(out_classes=classes, pretrained_model_name=pretrianed_model_name, pretrained=pretrained, ctx=ctx)
	net.hybridize()

	# 定义 Trainer
	trainer = gluon.Trainer(net.collect_params(),
	                        'sgd', {
		                        'learning_rate': lr, 'momentum': momentum, 'wd': wd
	                        })
	metric = mx.metric.Accuracy()
	# 定义准确率评估函数，损失函数
	L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True)

	best_val_acc, not_improve_count = 0.0, 0
	logging.info('start training, pretrained model_name:%s' % pretrianed_model_name)
	for epoch in range(epochs):
		tic = time.time()

		train_loss = 0
		metric.reset()
		num_batch = len(train_data)

		for index, batch in enumerate(train_data):
			# data [batch_size, 3, height, width], label [batch_size, ]
			data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
			label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
			# # mixup
			if MIX_UP:
				lam = np.random.beta(alpha, alpha)
				if epoch >= epochs - 20:
					lam = 1

				mix_data = [lam * X + (1 - lam) * X[::-1] for X in data]
				mix_label = []
				for Y in label:
					y1 = label_transform(Y, classes)
					y2 = label_transform(Y[::-1], classes)
					mix_label.append(lam * y1 + (1 - lam) * y2)

				data = mix_data
				label = mix_label

			with ag.record():
				outputs = [net(X) for X in data]
				loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
			for l in loss:
				l.backward()

			trainer.step(batch_size)
			train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

			metric.update(label, outputs)
			# 显示进度
			progressbar(index, num_batch - 1)

		# 计算 train 的评价标准
		_, train_acc = metric.get()
		train_loss /= num_batch
		# 计算 val 的评价标准
		val_acc, val_loss = validate(net, val_data, ctx)
		logging.info(
			'[Epoch %d] Train-acc: %.6f, loss: %.6f | Val-acc: %.6f, loss: %.6f | time: %.1f | lr: %f' %
			(epoch, train_acc, train_loss, val_acc, val_loss, time.time() - tic, trainer.learning_rate))
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			not_improve_count = 0
			logging.info('better result, saveing model, acc:%.4f' % best_val_acc)
			net.collect_params().save(model_weight_save_path)
		else:
			not_improve_count += 1
			if not_improve_count > MAX_IMPROVE_COUNT:
				not_improve_count = 0
				trainer.set_learning_rate(trainer.learning_rate * (0.1 ** 0.5))
				logging.info('reduce learning rate:%f' % trainer.learning_rate)
