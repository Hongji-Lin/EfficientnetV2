import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import efficientnetv2_s as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # load image
    test_path = '/home/lhj/PycharmProjects/EfficientnetV2/data/train'
    # 遍历文件夹，一个文件夹对应一个类别
    bin_class = [cla for cla in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, cla))]
    bin_class.sort()
    # 遍历每个文件夹下的文件
    for cla in bin_class:
        cla_path = os.path.join(test_path, cla)
        img_path = [os.path.join(cla_path, i) for i in os.listdir(cla_path)]
        print(img_path)
        for img_path in img_path:
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # read class_indict
            json_path = './class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            with open(json_path, "r") as f:
                class_indict = json.load(f)

            # create model
            model = create_model(num_classes=2).to(device)
            # load model weights
            model_weight_path = "/home/lhj/PycharmProjects/EfficientnetV2/weights/2022-11-17_model_best.pth"
            model.load_state_dict(torch.load(model_weight_path, map_location=device))
            model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            plt.title(print_res)
            for i in range(len(predict)):
                print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                          predict[i].numpy()))
            plt.show()


if __name__ == '__main__':
    main()
