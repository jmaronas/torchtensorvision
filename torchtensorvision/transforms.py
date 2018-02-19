import torch
import Functional as F
import numpy
import random

seed_flag=0

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class ToGPU(object):
	def __call__(self,img):
		if not img.is_cuda:
			img=img.cuda()
		return img

class RandomHorizontalFlip(object):
	''' Horizontal fliping'''
	def __init__(self,shape,seed=1):
		self.aux=torch.cuda.LongTensor(list(reversed(range(shape))))
		self.shape=shape
		global seed_flag
		if not seed_flag:
			random.seed(a=seed)
			seed_flag=True

	def __call__(self,img):
		assert self.shape==img.shape[-1]
		if random.random()<0.5:
			img = F.hflip(img,self.aux)
		return img

class RandomCrop(object):
	'''Cropping a value. Only square supported'''
	def __init__(self,cropsize,seed=1):
		global seed_flag
		self.shape=cropsize
		if not seed_flag:
			random.seed(a=seed)
			seed_flag=True

	def get_shapes(self,img):
		_,w, h = img.shape
		th, tw = self.shape,self.shape
		i = random.randint(0, h - th -1)
		j = random.randint(0, w - tw -1)
		return i,j,th,tw

	def __call__(self,img):

		i,j,th,tw=self.get_shapes(img)
		return F.Crop(img,i,j,th,tw)	
	
class CentralCrop(object):
	'''Cropping a value. Only square supported'''
	def __init__(self,cropsize):
		self.shape=cropsize

	def get_shapes(self,img):
		_,w, h = img.shape
		th, tw = self.shape,self.shape
		i = int(round((h-th)/2.))
		j = int(round((w-tw)/2.))
		return i,j,th,tw

	def __call__(self,img):

		i,j,th,tw=self.get_shapes(img)
		return F.Crop(img,i,j,th,tw)	

class Zscore(object):
	'''0-mean 1-std normalization'''
	def __call__(self,img,mean=[0,0,0],std=[1,1,1]):
		for c in len(img.shape):
			img[c]=F.Zscore(img[c],mean[c],std[c])
		return img
		

