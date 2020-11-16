import torchvision.models as models
import torch
import numpy as np
from functools import partial
import time

imsize = (1000, 2000)
loadnp = partial(np.fromfile, dtype=np.float32)

m = models.vgg16()
c = m.features._modules['0'].requires_grad_(False)
#n = torch.rand(20, 3, *imsize)
n = torch.from_numpy(loadnp('input.dat').reshape(20, 3, *imsize))

c.weight.numpy().tofile('weight.dat')
c.bias.numpy().tofile('bias.dat')
#n.numpy().tofile('input.dat')

print('start timing', time.ctime())
t1 = time.time()
n2 = c(n)
print('end timing', time.ctime())
print(time.time() - t1)

n2 = n2[0, :, 1:999, 1:1999].numpy()
n2.tofile('output.dat')

def diff():
    n3 = loadnp('output2.dat').reshape(20, 64, imsize[0]-2, imsize[1]-2)[0]
    return ((n3 - n2)**2).sum()

