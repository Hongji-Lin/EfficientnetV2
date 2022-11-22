# encoding: utf-8
# @author: Evan/Hongji-Lin
# @file: get_labels.py
# @time: 2022/11/18 上午12:16
# @desc:
import json
import os


def main():
    test_path = '/home/lhj/PycharmProjects/EfficientnetV2/data/data_8934'
    # img_idx_path = ''
    # cla_idx_path = ''
    # json_label_path
    txt_file = open(test_path + '.txt', 'w')

    # 遍历文件夹，一个文件夹对应一个类别
    bin_class = [cla for cla in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, cla))]
    bin_class.sort()

    # 支持的文件后缀类型
    supported = ['.jpg', '.png', '.jpeg','.JPG', '.PNG', '.JPEG']

    # 获得类别索引文件，并将k,v -> v,k
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "not found {}".format(json_label_path)
    json_file = open(json_label_path, 'r')
    # {"0": "empty"}
    cla_idx = json.load(json_file)
    # {"empty": "0"}
    cla_idx = dict((v, k) for k, v in cla_idx.items())

    # 遍历每个文件夹下的文件
    for cla in bin_class:
        cla_path = os.path.join(test_path, cla)
        # 遍历获取supported支持的所有文件路径
        images = [img_name for img_name in os.listdir(cla_path)
                  if os.path.splitext(img_name)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()

        # 获取该类别对应的索引
        image_class = cla_idx[cla]

        for img in images:
            txt_file.write(os.path.join(cla_path, img) + ' ' + str(image_class))
            txt_file.write('\n')
    txt_file.close()


if __name__ == '__main__':
    main()