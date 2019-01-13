import math
import os
#import os.path
import sys
#from random import randint
from PIL import Image

def add_background(foreground, background):

	if foreground.size != background.size:
		background = background.resize(foreground.size)
	background.paste(foreground, (0, 0), foreground)
	return background


def add_backgrounds(fg_path, bg_path):

	background = Image.open(bg_path)
	foreground = Image.open(fg_path)
	img = add_background(foreground, background)
	img.save('out2.jpg')
		
#-----------------------------------
if __name__ == '__main__':
	
	fg_path = '2.png'
	bg_path = 'bg1024.jpg'

	#in_dir = in_dir.rstrip('/')
	#out_dir = out_dir.rstrip('/')
	#os.system('mkdir -p {0}'.format(out_dir))

	add_backgrounds(fg_path, bg_path)