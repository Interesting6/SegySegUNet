import matplotlib.pyplot as plt
import numpy as np
import segyio
import os

data_dir = "/home/cym/Datasets/StData-12/F3_block/"
data_path = os.path.join(data_dir, "F3Seis_IL190_490_Amplitude.segy")
label_path = os.path.join(data_dir, "F3Seis_IL190_490_Label.segy")
data_cube = np.transpose(segyio.cube(data_path), (0,2,1))
label_cube = np.transpose(segyio.cube(label_path), (0,2,1))

label2 = label_cube[1]
print(label2[:50, :10])
print("-------")
print(label2[:50, -10:])
