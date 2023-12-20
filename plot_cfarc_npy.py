import numpy as np
from PIL import Image
import glob

root = '/Data/federated_learning/kd_jetson_clip_ex/data/cifar10_c/CIFAR-10-C/'
pixelates = root+'pixelate.npy'
pixelate_np = np.load(pixelates)[254,:,:,:]
print(pixelate_np.shape)
image = Image.fromarray(pixelate_np, 'RGB')
image.show()