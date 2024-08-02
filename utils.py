import dill
import torch
import torch.nn as nn
from torch.utils.data import random_split

from taskVetcor import TaskVector


# 灰度图像转RGB
def convert2RGB(image):
    if image.mode == "L":
        image = image.convert("RGB")
    return image


# 根据数据集标签数修改网络分类头数
def change_classification_heads(model, in_features, new_num_classes):
    new_fc = nn.Linear(in_features=in_features, out_features=new_num_classes)
    model.fc = new_fc
    return model


num_classes_dict = {
    "PathMNIST": 9,
    "ChestMNIST": 14,
    "DermaMNIST": 7,
    "OCTMNIST": 4,
    "PneumoniaMNIST": 2,
    "RetinaMNIST": 5,
    "BreastMNIST": 2,
    "BloodMNIST": 8,
    "TissueMNIST": 8,
    "OrganMNIST": 11,
}


def update_log(log_path, content):
    with open(log_path, 'a', encoding='utf-8') as file:
        file.write(content)


def check_data(dataset, num_classes):
    label_count = {}
    counts = []

    for i in range(num_classes):
        label_count[i] = 0

    for data, label in dataset:
        label_count[label[0]] += 1

    for key in label_count:
        print(f"label: {key}, count: {label_count[key]}")
        counts.append(label_count[key])
    return counts


def split_dataset(dataset, split_num):
    indices = list(range(len(dataset)))
    split_nums = [1.0 / split_num] * split_num
    split_nums = [int(x * len(dataset)) for x in split_nums]

    # 保证所有元素之和==数据集长度
    split_nums[-1] = len(dataset) - (sum(split_nums) - split_nums[-1])

    splited_datasets = random_split(dataset, split_nums)
    for elem in splited_datasets:
        print(len(elem))

    return splited_datasets


def save_dataset(dataset, path):
    with open(path, 'wb') as f:
        dill.dump(dataset, f)


def load_dataset(path):
    with open(path, 'rb') as f:
        return dill.load(f)

"""
def merge_taskVec(taskVec_list):
    # 任务向量相加
    avg_state_dict={}
    #初始化
    for key in taskVec_list[0].keys():
        if taskVec_list[0][key].dtype in [torch.int64,torch.uint8]:
            avg_state_dict[key]=taskVec_list[0][key]
            continue

        avg_state_dict[key]=torch.zeros_like(taskVec_list[0][key])

        



    return TaskVector(vector=new_vector)
"""