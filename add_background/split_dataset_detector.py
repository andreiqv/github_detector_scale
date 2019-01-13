import os
import sys
import random

if os.path.exists('.local'):
	src_dir = '/mnt/lin2/_test/in/'
	dst_dir = '/mnt/lin2/_test/splited/'
else:
	src_dir = '/home/andrei/Data/Datasets/ScalesDetector/detector-261018/'
	dst_dir = '/home/andrei/Data/Datasets/ScalesDetector/dataset-splited/'


parts = ['train', 'valid', 'test']


def split_copy_files(src_dir, dst_dir, parts, ratio=[1,1,1]):

	src_dir = src_dir.rstrip('/')
	dst_dir = dst_dir.rstrip('/')

	os.system('mkdir -p {}'.format(dst_dir))
	for p in parts:
		os.system('mkdir -p {}'.format(dst_dir + '/' + p))

	file_names = os.listdir(src_dir)
	if len(file_names) == 0: 
		print('{0} - empty subdir'.format(class_name))
		return 0

	file_names = list(filter(lambda s: s[-4:]=='.jpg', file_names))
		
	# calculate train, valid and test sizes
	num_files = len(file_names)
	num_valid = num_files * ratio[1] // sum(ratio)
	num_test  = num_files * ratio[2] // sum(ratio)
	num_train = num_files - num_valid - num_test

	# SHUFFLE OR SORT
	random.shuffle(file_names)
	#file_names.sort()
		
	files = dict()
	files['train'] = file_names[ : num_train]
	files['valid'] = file_names[num_train : num_train + num_valid]
	files['test']  = file_names[num_train + num_valid : ]
	print('{} - {}:{}:{}'.format(num_files, num_train, num_valid, num_test))

	for part in parts:
		cmd = 'mkdir -p {}'.format(dst_dir + '/' + part)
		os.system(cmd)

		for name in files[part]:
			base_name = name[:-4]
			#print(base_name)
			exts = ['jpg', 'txt', 'jpg.json', 'png']
			for ext in exts:
				file_name = base_name + '.' + ext
				src_path = src_dir + '/' + file_name
				dst_path = dst_dir + '/' + part + '/' + file_name
				cmd = 'cp {} {}'.format(src_path, dst_path)
				print(cmd)
				os.system(cmd)

if __name__ == '__main__':

	split_copy_files(src_dir, dst_dir, parts, ratio=[85,15,0])
	#split_copy_files(src_dir, dst_dir, parts, ratio=[1,1,0])

