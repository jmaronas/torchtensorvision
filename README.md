TORCHTENSORVISION
This package is intended to do the same as torchvision but to directly operate on torch tensor. This makes possible to do operations directly on GPU and not to deal with PIL options for images. For example you can process image in float32 directly instead of using uint8 as PIL. 
Other example is you can do operations in the order you want. For example you maybe prefer normalizing data distribution without reescaling to range 0-1...
