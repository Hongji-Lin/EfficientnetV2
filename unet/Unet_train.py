# encoding: utf-8
# @author: Evan/Hongji-Lin
# @file: Unet_train.py
# @time: 2022/11/25 下午4:48
# @desc:
import argparse
import math
import os
import time
import datetime

import torch
from torch import optim
from torch.optim import optimizer, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from train_utils.my_dataset import MyDataSet
from src import UNet
from train_utils.utils import read_split_data, train_one_epoch, evaluate, plot_data_loader_image, plot_class_preds


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    print(device)
    tb_writer = SummaryWriter()
    if os.path.exists("../unet/weights") is False:
        os.makedirs("../unet/weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop([224, 224]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize([224, 224]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=8,
                                               collate_fn=train_dataset.collate_fn)
    train_imgs, _ = next(iter(train_loader))
    print("train_imgs_shape:{}".format(train_imgs.shape))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=8,
                                             collate_fn=val_dataset.collate_fn)
    val_imgs, _ = next(iter(train_loader))
    print("val_imgs_shape:{}".format(val_imgs.shape))

    # 如果存在预训练权重则载入
    model = UNet(in_channels=3, num_classes=2, bilinear=True, base_c=16).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    min_loss = 100000  # 随便设置一个比较大的数
    for epoch in range(args.epochs):
        # train
        start = time.time()
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # # 显示每个epoch的train_loader图片
        # input_img = plot_data_loader_image(train_loader)
        # tb_writer.add_figure("Input Images",
        #                      figure=input_img,
        #                      global_step=epoch)
        #
        # # add figure into tensorboard
        # fig = plot_class_preds(model=model, data_loader=val_loader, device=device)
        #
        # tb_writer.add_figure("predictions vs. actuals",
        #                      figure=fig,
        #                      global_step=epoch)

        # save model
        time_str = time.strftime('%Y-%m-%d_')
        if val_loss < min_loss:
            min_loss = val_loss
            print("save model")
            weights_savepath = "../unet/weights" + time_str + "model_best.pth"
            torch.save(model.state_dict(), weights_savepath)
            print("最好的模型在：epoch = {}".format(epoch))

        end = time.time()
        print("每个epoch训练的时间为：{}".format(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str, default="/home/binoverflow/EfficientnetV2/data/data_unet_test")
    parser.add_argument('--weights', type=str, default=None, help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    # parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1 or cpu)') # 這一句專門在只有cpu的電腦上執行

    opt = parser.parse_args()

    main(opt)
