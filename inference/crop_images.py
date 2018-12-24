import math
import os
import os.path
import sys
from random import randint
from PIL import Image

def crop_images(in_dir, out_dir):

	files = os.listdir(in_dir)
	
	for file_name in files:
		print(file_name)
		file_path = in_dir + '/' + file_name

		#path_base = os.path.splitext(file_path)[0]
		ext = os.path.splitext(file_name)[1]
		if ext != '.jpg':
			continue
		
		txt_file = os.path.splitext(file_path)[0] + '.txt'
		#print(txt_file)
		f = open(txt_file, 'rt')
		line = f.readline().rstrip().split()
		f.close()
		x, y, w, h = [float(t) for t in line]
		if x == y == w == h == 0:
			continue
		print(x, y, w, h)

		img = Image.open(file_path)
		img = img.resize((299, 299))
		sx, sy = img.size
		print('sx={0}, sy={1}'.format(sx,sy))
		x1 = int((x - w/2) * sx)
		x2 = int((x + w/2) * sx)
		y1 = int((y - h/2) * sy)
		y2 = int((y + h/2) * sy)
		area = (x1, y1, x2, y2)
		print(area)
		box = img.crop(area)
		out_file = out_dir + '/' + file_name + '_crop.jpg'
		box.save(out_file)

		img.close()
		
#-----------------------------------
if __name__ == '__main__':
	
	in_dir = '/home/andrei/Data/Datasets/ScalesDetector/detector-261018/'
	out_dir = '/home/andrei/Data/Datasets/ScalesDetector/_tmp/'
	#in_dir = '/w/WORK/ineru/06_scales/images/'
	#out_dir = '/w/WORK/ineru/06_scales/images/out/'
	

	in_dir = in_dir.rstrip('/')
	out_dir = out_dir.rstrip('/')
	os.system('mkdir -p {0}'.format(out_dir))

	crop_images(in_dir, out_dir)