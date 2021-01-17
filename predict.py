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


if __name__ == "__main__":
    print("main")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    patch_size = 64
    batch_size = 8
    num_class = 13
    save_dir = "./results"
    # 加载数据集
    data_dir = "/home/cym/Datasets/StData-12/F3_block/"
    dataset = F3DS(data_dir, ptsize=patch_size, train=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)
    hw = dataset.hw
    origin_mask = np.zeros((hw[0], hw[1]))

    net = UNet(n_channels=1, n_classes=num_class, bilinear=False)
    net.load_state_dict(torch.load('models2/best_model.pth'))
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
            pred_label = pred_label.cpu().numpy() / 12
            for img,lb,plb,hy,wx in zip(image, label, pred_label,hys,wxs):
                img_path = f"./results2/img/img_{img_idx}.png"
                lb_path = f"./results2/label/lb_{img_idx}.png"
                plb_path = f"./results2/predlabel/plb_{img_idx}.png"
                plt.imsave(img_path, np.squeeze(img, 0), cmap="gray")
                plt.imsave(lb_path, lb, cmap="gray")
                plt.imsave(plb_path, plb, cmap="gray")

                origin_mask[hy:hy+patch_size, wx:wx+patch_size] = plb

                img_idx += 1
    
        print('Test Progress: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
           batch_idx, all_images_num, len(test_loader.dataset),
                100. * (batch_idx + 1) / len(test_loader), 
                loss.item(), correct*100))
    print("inference done!")
    all_acc = all_acc / all_images_num
    print(f"average accuracy = {all_acc}")
    pred_label_path = "./results2/pred_label.png"
    plt.imsave(pred_label_path, origin_mask, cmap="gray")
    # origin_mask = (origin_mask * 255).astype("uint8")
    # origin_mask = Image.fromarray(origin_mask)
    # origin_mask.save("results2/pred_label2.png")


