import glob
import numpy as np
import torch
from torch import nn
import os
# import cv2
from utils.dataset2 import F3DS
from model.unet_model import UNet
from matplotlib import pyplot as plt
from PIL import Image

from imgaug.augmentables.segmaps import SegmentationMapsOnImage



def GaryTo3C(image):
    image = np.expand_dims(image, axis=2)
    image = np.repeat(image, 3, axis=2)
    if image.dtype.kind == 'f':
        image = f2uint8(image)
    return image

def LabelTo3C(labelimg, n_class=12):
    labelimg = np.expand_dims(labelimg, axis=2)
    labelimg = np.repeat(labelimg, 3, axis=2)
    labelimg = (labelimg).astype("uint8")
    return labelimg

def f2uint8(image):
    return (image*255).astype("uint8")


if __name__ == "__main__":
    print("main")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    patch_size = 128
    batch_size = 8
    num_class = 13
    save_dir = "./results128+20"
    # 加载数据集
    data_dir = "/home/cym/Datasets/StData-12/F3_block/"
    dataset = F3DS(data_dir, ptsize=patch_size, train=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)
    hw = dataset.hw
    pred_mask = np.zeros((hw[0], hw[1]))

    data2 = GaryTo3C(dataset.data2)
    label2 = dataset.label2.astype("uint8")
    print(np.unique(label2))
    print(label2.shape)

    net = UNet(n_channels=1, n_classes=num_class, bilinear=False)
    net.load_state_dict(torch.load('models128+20/last_model.pth'))
    net.to(device=device)
    net.eval()
    print("net prepare done")
    if not os.path.exists(save_dir):
        os.makedirs(f"{save_dir}/img")
        os.makedirs(f"{save_dir}/label")
        os.makedirs(f"{save_dir}/predlabel")

    all_images_num = 0.
    all_acc = 0.

    img_idx = 0
    for batch_idx, (image, label, hys, wxs) in enumerate(test_loader):

        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device)
        batch_len = len(image)
        
        with torch.no_grad():
            pred = net(image)
            label = label.squeeze(1).long()
            loss = criterion(pred, label)
            pred = torch.softmax(pred, dim=1)
            _, pred_label = pred.max(dim=1)
            correct = label == pred_label
            correct = correct.sum().item() / correct.numel()

            all_acc += correct * batch_len
            all_images_num += batch_len

            # save image
            image = image.cpu().numpy()
            label = label.cpu().numpy()
            pred_label = pred_label.cpu().numpy() # / 12
            for img,lb,plb,hy,wx in zip(image, label, pred_label,hys,wxs):
                # img_path = f"./results2/img/img_{img_idx}.png"
                # lb_path = f"./results2/label/lb_{img_idx}.png"
                # plb_path = f"./results2/predlabel/plb_{img_idx}.png"
                # plt.imsave(img_path, np.squeeze(img, 0), cmap="gray")
                # plt.imsave(lb_path, lb, cmap="gray")
                # plt.imsave(plb_path, plb, cmap="gray")

                pred_mask[hy:hy+patch_size, wx:wx+patch_size] = plb

                img_idx += 1
    
        print('Test Progress: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
           batch_idx, all_images_num, len(test_loader.dataset),
                100. * (batch_idx + 1) / len(test_loader), 
                loss.item(), correct*100))
    print("inference done!")
    all_acc = all_acc / all_images_num
    print(f"average accuracy = {all_acc}")

    pred_label_path = f"{save_dir}/pred_label.png"
    plt.imsave(pred_label_path, pred_mask, cmap="gray")

    # 将label涂到原image上
    segmap = SegmentationMapsOnImage(label2, data2.shape)
    data2_wlabel2 = segmap.draw_on_image(data2, alpha=0.4, draw_background=True)[0]
    save_path = f"{save_dir}/data_with_label.png"
    plt.imsave(save_path, data2_wlabel2)

    # 将pred_mask涂到原image上
    pred_mask = pred_mask.astype("uint8")
    segmap = SegmentationMapsOnImage(pred_mask, data2.shape)
    data2_wpred = segmap.draw_on_image(data2, alpha=0.4, draw_background=True)[0]
    save_path = f"{save_dir}/data_with_predmask.png"
    plt.imsave(save_path, data2_wpred)


