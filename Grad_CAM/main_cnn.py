import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from model import efficientnetv2_s as create_model
import argparse


def main(args):
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # model = create_model(num_classes=args.num_classes)
    # if args.weights != "":
    #     if os.path.exists(args.weights):
    #         weights_dict = torch.load(args.weights, map_location=device)
    #         load_weights_dict = {k: v for k, v in weights_dict.items()
    #                              if model.state_dict()[k].numel() == v.numel()}
    #         print(model.load_state_dict(load_weights_dict, strict=False))
    #     else:
    #         raise FileNotFoundError("not found weights file: {}".format(args.weights))
    # target_layers = [model.blocks[-1]]

    # model = create_model(num_classes=args.num_classes)
    # model.load_state_dict(torch.load("../weights/pre_efficientnetv2-s.pth"))
    # model = model.to(device)
    # model.eval()
    # target_layers = [model.blocks[-1]]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # create model
    model = create_model(num_classes=1000).to(device)
    # load model weights
    model_weight_path = "../weights/pre_efficientnetv2-s.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device), strict=False)
    model.eval()
    target_layers = [model.blocks[-1]]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    img_path = "full.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 1 # 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--weights', type=str, default='../weights/pre_efficientnetv2-s.pth',
                        help='initial weights path')
    opt = parser.parse_args()
    main(opt)
