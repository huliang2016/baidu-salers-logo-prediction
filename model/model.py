# -*- coding: UTF-8 -*-
'''
Created on 2018/5/1
@author: ByRookie
'''
import mxnet as mx
import mxnet.gluon.model_zoo.vision as model_zoo
from mxnet.gluon import nn
from mxnet.initializer import Xavier


class Model(nn.HybridBlock):
	def __init__(self, out_classes=100, pretrained_model_name='resnet50_v2', pretrained=True, ctx=mx.cpu(0), **kwargs):
		super(Model, self).__init__(**kwargs)
		self.feature_name = pretrained_model_name
		pretrained_model = model_zoo.get_model(pretrained_model_name, pretrained=pretrained, ctx=ctx)
		# feature extractor
		with self.name_scope():
			self.feature_extractor = nn.HybridSequential()
			for layer in pretrained_model.features[:-1]:
				self.feature_extractor.add(layer)
			# init feature_extractor params
			if not pretrained:
				self.feature_extractor.collect_params().initialize(init=Xavier(), ctx=ctx)

			self.output = nn.HybridSequential()
			# self.output.add(MyFlatten())
			# self.output.add(nn.Conv1D(256, 1))
			self.output.add(nn.Flatten())
			self.output.add(nn.Dense(512, activation='relu'))
			self.output.add(nn.Dropout(0.5))
			# self.output.add(nn.Dense(256, activation='relu'))
			# self.output.add(nn.Dropout(0.5))
			self.output.add(nn.Dense(out_classes))
			# init output params
			self.output.collect_params().initialize(init=Xavier(), ctx=ctx)

	def hybrid_forward(self, F, x, *args):
		features = self.feature_extractor(x)
		f_cls = self.output(features)
		return f_cls


class MyFlatten(nn.HybridBlock):
	def __init__(self, **kwargs):
		super(MyFlatten, self).__init__(**kwargs)

	def hybrid_forward(self, F, x, *args, **kwargs):
		return x.reshape((0, 0, -1))
