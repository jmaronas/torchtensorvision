import torch

def hflip(img,aux):
	return torch.index_select(img,2,aux)

def Crop(img,i,j,th,tw):
	return img[:,i:i+th,j:j+tw]

def Zscore(img,mean,std):
		return (img-mean)/std
