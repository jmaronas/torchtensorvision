# TORCHTENSORVISION
The aim of this package is to do exactly the same as torchvision but to directly operate on tensors. This makes possible to do operations directly on GPU, and avoid the overhead of transfering images to GPU on each epoch. My intention is not to replicate the whole package but to implement the functions I find usefull and upload them here. The usage of this package is the same as in torchvision version 0.3.0 [torchvision](https://pytorch.org/docs/0.3.1/)

Just create your data loader as specified in torch, but now you can held your data in a torch.cuda or torch tensor and apply directly this transformations. Basically when you create your dataloader in torchvision a location of your data. If for example your data lies in a numpy array you have to first convert to PIL and then apply transformations. In this case you can directly hold your data on GPU and apply directly the transformations.

## Available Functionality
### Same as in torchvision
Compose

RandomCrop

CentralCrop

Zscore

Rotate

### Added by me
ToGPU: move batch to gpu

RandomHorizontalFlip: Is available but you have to pass as argument the shape of your image.

## INSTALL
You can install either in ./local or in a virtual enviroment with pip. Just type:
```pip install git+https://github.com/jmaronas/torchtensorvision.git```

## TODO
Change Random Horizontal Flip to work on gpu

Upload version

Rotate is the same as torchvision as there is no rotate function for a torch tensor for the moment
