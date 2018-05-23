# -*- coding: UTF-8 -*-
'''
Created on 2018/5/1
@author: ByRookie
'''
import os
import random
import shutil

if __name__ == '__main__':

	random.seed(2018)
	base_dir = '../datasets'

	train_file = '../datasets/train.txt'
	train_dir = '../datasets/train/'
	val_dir = '../datasets/valid/'

	# get train files
	train_class2Image = {}
	with open(train_file, 'r') as file:
		contents = file.readlines()
		for content in contents:
			content = content.replace('\n', '')
			content = content.split(' ')
			file, classes = content[0], content[1]
			if classes not in train_class2Image.keys():
				train_class2Image[classes] = []
			train_class2Image[classes].append(file)

	for index, key in enumerate(train_class2Image.keys()):
		print('%d/%d, dir:%s' % (index, len(train_class2Image.keys()), key), end='\r')

		dirPath = os.path.join(train_dir, key)
		valPath = os.path.join(val_dir, key)
		# make train dir
		if not os.path.exists(dirPath):
			os.makedirs(dirPath)
		# make val dir
		if not os.path.exists(valPath):
			os.makedirs(valPath)
		# move file to dir
		for train_image in train_class2Image[key]:
			origin_path = os.path.join(train_dir, train_image)
			if random.random() > 0.2:
				dest_path = os.path.join(dirPath, train_image)
			else:
				dest_path = os.path.join(valPath, train_image)
			shutil.move(origin_path, dest_path)
