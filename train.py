# -*- coding: utf-8 -*-  
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np

from config import config as cfg  # 导入配置参数
from src.dataloader import yolo_dataset_collate, YoloDataset
from model.yolo3 import YoloBody
from model.yolo_training import YOLOLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, yolo_losses, optimizer, train_loader, epoch, cfg):
    model.train()
    total_loss = 0.
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).to(device)
        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
        optimizer.zero_grad()
        outputs = model(images)
        losses = []
        for i in range(3):
            loss_item = yolo_losses[i](outputs[i], targets)
            losses.append(loss_item[0])
        loss = sum(losses)
        loss.backward()
        optimizer.step()

        total_loss += loss

        if (batch_idx + 1) % cfg.interval == 0:
            print("[{}/{}]|[{}/{}]|loss:{:.4f}".format(epoch, cfg.num_epochs,
                                                  batch_idx+1, len(train_loader)+1,
                                                  total_loss / (batch_idx + 1)))
            total_loss = 0.

def test(model, yolo_losses, test_loader):
    model.eval()
    test_loss = 0.
    for batch_idx, (images, targets) in enumerate(test_loader):
        with torch.no_grad():
            images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).to(device)
            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            outputs = model(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            test_loss += loss

    print("test loss:{:.4f}".format(test_loss/ len(test_loader)))


def main():
    # 输出
    if not os.path.exists(cfg.output):
        os.mkdir(cfg.output)
    # 训练集
    train_set = YoloDataset(cfg.train_list, (cfg.img_h, cfg.img_w))
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
    # 测试集
    test_set = YoloDataset(cfg.test_list, (cfg.img_h, cfg.img_w))
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                             pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

    # 开始的批次
    start_epoch = cfg.start_epoch

    # 网络
    model = YoloBody(3, 20)
    # 预训练
    if cfg.pretrained != "":
        print("load pretrained model: {}".format(cfg.pretrained))
        start_epoch = int(cfg.pretrained.split("/")[-1].split(".")[2])
        model.load_state_dict(torch.load(cfg.pretrained))
    model.to(device)

    # 损失函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(cfg.anchors, [-1, 2]), cfg.classes, (cfg.img_h, cfg.img_h), cuda=True))

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        train(model, yolo_losses, optimizer, train_loader, epoch, cfg)
        if epoch % cfg.valinterval == 0:
            test(model, yolo_losses, test_loader)
            torch.save(model.state_dict(), f"./{cfg.output}/yolov3.voc.{epoch}.pt")


if __name__ == '__main__':
    main()
