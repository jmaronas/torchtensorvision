import torch
import numpy
from skimage.transform import rotate as imrotate

def hflip(img,aux):
	return torch.index_select(img,2,aux)

def Crop(img,i,j,th,tw):
	return img[:,i:i+th,j:j+tw]

def Zscore(img,mean,std):
		return (img-mean)/std


def rotate(img,angle):#unsupported the rest of torchvision options
	if img.is_cuda:
		img=img.cpu()
	img.numpy()

	img=numpy.transpose(img,(1,2,0))
	img=imrotate(img,angle,mode='symmetric')
	img=numpy.transpose(img,(2,0,1))
	return torch.from_numpy(img)
	
	
