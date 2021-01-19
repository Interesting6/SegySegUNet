from model.unet_model import UNet
from utils.dataset2 import F3DS
from torch import optim
import torch.nn as nn
from torch.nn.init import xavier_normal_, normal_
import torch.nn.functional as F
import torch
import os, copy
from tensorboardX import SummaryWriter

def train_net(net, data_ld, device, writer, epochs=100, lr=0.0001, save_dir="./models"):
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    best_loss = float('inf')
    
    batch_size = data_ld.batch_size
    image_size = data_ld.dataset.ptsize
    # 训练epochs次
    tensorboard_ind = 0
    for epoch in range(epochs):
        print(f"-----------{epoch}-----------")
        net.train()
        epoch_loss = 0
        epoch_img_num = 0
        epoch_acc = 0
        for batch_idx, (image, label) in enumerate(data_ld):
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device)
            logits = net(image)

            # logits = logits.permute(0, 2,3, 1).contiguous().view(-1, num_class)
            # label = label.permute(0, 2,3,1).contiguous().view(-1, 1)
            label = label.squeeze(1).long()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                img_num = len(image)
                epoch_loss += loss.item() * img_num
                epoch_img_num += img_num
                _, pred = F.softmax(logits, dim=1).max(dim=1)
                correct = (pred == label).view(img_num, -1)
                acc = correct.float().mean()
                epoch_acc += acc.item() * img_num

            if (batch_idx + 1) % 32 == 0:
                print('Train Epoch: {} [batch {}, dataloader sample: {}/{} ({:.0f}%)]\t; Batch: Loss: {:.6f}\tAcc: {:.6f}'.format(
                    epoch, batch_idx, (batch_idx + 1) * len(image), len(data_ld.dataset),
                        100. * (batch_idx + 1) / len(data_ld), loss.item(), acc.item()))

        epoch_loss = epoch_loss / epoch_img_num
        epoch_acc = epoch_acc / epoch_img_num
        print(f"epoch logs: \t epoch_loss={epoch_loss}, \t epoch_acc={epoch_acc}")

        writer.add_scalar("loss", epoch_loss, epoch)
        writer.add_scalar("accuray", epoch_acc, epoch)
        
        # 保存loss值最小的网络参数
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_loss_weights = copy.deepcopy(net.state_dict())
            
    print(f" {epoch} epoch trainning end!")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    torch.save(net.state_dict(), f'{save_dir}/last_model.pth')
    torch.save(best_loss_weights, f'{save_dir}/best_model.pth')
            


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_normal_(m.weight.data)
        normal_(m.bias.data)
    if isinstance(m, nn.ConvTranspose2d):
        xavier_normal_(m.weight.data)
        normal_(m.bias.data)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    patch_size = 128
    batch_size = 64
    num_class = 13
    epochs = 200
    # 加载数据集
    data_dir = "/home/cym/Datasets/StData-12/F3_block/"
    dataset = F3DS(data_dir, ptsize=patch_size, train=True)
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)

    # 加载网络，图片单通道1，分类为13。
    net = UNet(n_channels=1, n_classes=num_class, bilinear=False)
    net.apply(weight_init)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    writer = SummaryWriter('./logs128+20')
    
    train_net(net, train_loader, device, writer, epochs, save_dir="./models128+20")



