# 配置设备
import torch
import torchvision.models as models
from medmnist import BreastMNIST, DermaMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from params import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = num_classes_dict["DermaMNIST"]

transform = transforms.Compose([
    transforms.Lambda(convert2RGB),  # 统一为RGB格式
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集，medMNIST的子数据集
train_dataset = DermaMNIST(split="train", download=True, size=224, transform=transform, root="./data/")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = DermaMNIST(split="test", download=True, size=224, transform=transform, root="./data/")

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = change_classification_heads(model, in_features=512, new_num_classes=num_classes)
torch.save(model.cpu(),"./checkpoints/pretrained_checkpoint.pt")