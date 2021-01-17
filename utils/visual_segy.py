import numpy as np
import segyio
from PIL import Image
import os
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage
import torch

data_dir = "/home/cym/Datasets/StData-12/F3_block/"
data_path = os.path.join(data_dir, "F3Seis_IL190_490_Amplitude.segy")
label_path = os.path.join(data_dir, "F3Seis_IL190_490_Label.segy")
data_cube = np.transpose(segyio.cube(data_path), (0,2,1))
label_cube = np.transpose(segyio.cube(label_path), (0,2,1))


# # MATPLOTLIB
# out_path_tmp = "outs/data_{num}_img.png"
# for i in range(data_cube.shape[0]):
#     out_path = out_path_tmp.format(num=i)
#     plt.imsave(out_path, data_cube[i], cmap="gray")

# out_path_tmp = "outs/label_{num}_img.png"
# for i in range(label_cube.shape[0]):
#     out_path = out_path_tmp.format(num=i)
#     plt.imsave(out_path, label_cube[i], cmap="gray")


print(data_cube[0, :16, :16])

