# -*- coding: UTF-8 -*-
'''
Created on 2018/5/2
@author: ByRookie
'''
import pickle

import numpy as np

if __name__ == '__main__':
	model1Result = './datasets/resnet152_v2_pred_test.pickle'
	model2Result = './datasets/densenet161_pred_test.pickle'
	model3Result = './datasets/resnet50_v2_pred_test.pickle'
	model4Result = './datasets/inceptionv3_pred_test.pickle'
	# model1Result = './datasets/resnet152_v2_pred.pickle'
	# model2Result = './datasets/densenet161_pred.pickle'
	# model3Result = './datasets/resnet50_v2_pred.pickle'
	# model4Result = './datasets/inceptionv3_pred.pickle'

	# acc1 = 1
	# acc2 = 1
	# acc3 = 1
	# acc4 = 1
	acc1, acc2, acc3, acc4 = 0.9889, 0.9870, 0.9833, 0.9759
	# acc1, acc2, acc3, acc4 = 0.9889, 0.9870, 0.9833, 0.9759

	result1 = pickle.load(open(model1Result, 'rb'))
	result2 = pickle.load(open(model2Result, 'rb'))
	result3 = pickle.load(open(model3Result, 'rb'))
	result4 = pickle.load(open(model4Result, 'rb'))

	sorted_ids = list(range(1, 101))
	sorted_ids.sort(key=lambda x: str(x))
	# sorted_ids.remove(13)

	# result_file = './datasets/resnet152_v2_0.9703_densenet161_0.9647_inceptionV3_0.9351.txt'
	result_file = './pred/resnet152_v2_0.9889_densenet161_0.9870_resnet50_v2_0.9833_inceptionv3_0.9759.csv'

	right = 0
	# 预测过程结果
	with open(result_file, 'w') as file:
		for key in result2.keys():
			# 		# 投票
			result1[key] = result1[key] >= max(result1[key])
			result2[key] = result2[key] >= max(result2[key])
			result3[key] = result3[key] >= max(result3[key])
			result4[key] = result4[key] >= max(result4[key])
			temp = np.asarray(acc1 * result1[key] + acc2 * result2[key] + acc3 * result3[key] + acc4 * result4[key])
			file.write('%s %s\n' % (key.split('/')[-1], sorted_ids[np.argmax(temp)]))
# idx = int(key.split('/')[-2])
# if idx == sorted_ids[np.argmax(temp)]:
# 	right += 1
# else:
# 	print(key, idx, sorted_ids[np.argmax(temp)], sorted_ids[np.argmax(result1[key])],
# 	      sorted_ids[np.argmax(result2[key])], sorted_ids[np.argmax(result3[key])],
# 	      sorted_ids[np.argmax(result4[key])])
# sorted_ids[np.argmax(result3[key])], )
# sorted_ids[np.argmax(result4[key])])
#
# print('%d/%d, %.4f' % (right, len(result2.keys()), float(right) / len(result2.keys())))
