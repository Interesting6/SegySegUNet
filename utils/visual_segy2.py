import numpy as np
import segyio
from PIL import Image
import os
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torch
from torchvision import transforms

data_dir = "/home/cym/Datasets/StData-12/F3_block/"
data_path = os.path.join(data_dir, "F3Seis_IL190_490_Amplitude.segy")
label_path = os.path.join(data_dir, "F3Seis_IL190_490_Label.segy")
data_cube = np.transpose(segyio.cube(data_path), (0,2,1))
label_cube = np.transpose(segyio.cube(label_path), (0,2,1))



# min4  = label_cube.min(axis=(1,2))
# print(min4)
# label_cube = label_cube - np.expand_dims(min4, axis=(1,2))
# print(label_cube.min(axis=(1,2)))
# range4 = label_cube.max(axis=(1,2)) - label_cube.min(axis=(1,2))
# range4 = np.expand_dims(range4, axis=(1,2))
# label_cube = (label_cube / range4 * 255).astype(np.uint8)

# out_path_tmp = "outs/img_data_{num}_img.png"
# for i in range(data_cube.shape[0]):
#     out_path = out_path_tmp.format(num=i)
#     img = Image.fromarray(data_cube[i], mode="L")
#     img.save(out_path)

# out_path_tmp = "outs/img_label_{num}_img.png"
# for i in range(label_cube.shape[0]):
#     out_path = out_path_tmp.format(num=i)
#     img = Image.fromarray(label_cube[i], mode="L")
#     img.save(out_path)


# ToPILImage
min4  = data_cube.min(axis=(1,2))
print(min4)
data_cube = data_cube - np.expand_dims(min4, axis=(1,2))
print(data_cube.min(axis=(1,2)))
range4 = data_cube.max(axis=(1,2))
range4 = np.expand_dims(range4, axis=(1,2))
data_cube = data_cube / range4
# data_cube = data_cube / range4

tpi = transforms.ToPILImage()
img = data_cube[0]
imgp = tpi(img)


data_cube2 = np.zeros_like(data_cube)
path_tmp = "outs/data_{num}_img.png"
for num in range(4):
    data_cube2[num] = plt.imread(path_tmp, cmap="gray")


data_1 = data_cube[1]
data_1_ = data_cube2[1]

correct = data_1

# data1 = data_cube[1]
# label1 = label_cube[1]
# print(data1.min(), data1.max())
# print(label1.min(), label1.max())
# data1p = Image.fromarray(data1, mode="F").convert("L")
# data1pa = np.array(data1p)
# print(data1pa.min(), data1pa.max(), np.unique(data1pa))

# data1int8 = data1.astype(np.uint8)
# print(data1int8.min(), data1int8.max())
# print()


# MATPLOTLIB
# out_path_tmp = "outs/plt_data_{num}_img.png"
# for i in range(data_cube.shape[0]):
#     out_path = out_path_tmp.format(num=i)
#     plt.imsave(out_path, data_cube[i], cmap="gray")

# out_path_tmp = "outs/plt_label_{num}_img.png"
# for i in range(label_cube.shape[0]):
#     out_path = out_path_tmp.format(num=i)
#     plt.imsave(out_path, label_cube[i], cmap="gray")

