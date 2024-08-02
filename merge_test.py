from collections import OrderedDict

import torch
from medmnist import DermaMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.models as models
from params import *

from utils import *

# 固定参数配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = num_classes_dict["DermaMNIST"]
num_dataset = 5

transform = transforms.Compose([
    transforms.Lambda(convert2RGB),  # 统一为RGB格式
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = DermaMNIST(split="test", download=True, size=224, transform=transform, root="./data/")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

params_list = []

for i in range(num_dataset):
    model = torch.load(f"./checkpoints/fintuned_checkpoint_{i + 1}.pt")
    params_list.append(model.state_dict())

avg_state_dict = OrderedDict()
for key in params_list[0].keys():
    avg_state_dict[key] = torch.zeros_like(params_list[0][key])
    if params_list[0][key].dtype in [torch.int64, torch.uint8]:
        continue

    for i in range(num_dataset):
        avg_state_dict[key] += params_list[i][key] / float(num_dataset)

avg_model = models.resnet18(weights=None)
avg_model = change_classification_heads(avg_model, in_features=512, new_num_classes=num_classes)
avg_model.load_state_dict(avg_state_dict)

# avg_model = avg_model.eval()
avg_model = avg_model.to(device)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device=device)
        labels = labels.reshape(-1).to(torch.int64).to(device=device)

        outputs = avg_model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    update_log(
        log_path="./logs/merge_log.txt",
        content=f"Avg model :Accuracy of the network on the {len(test_dataset)} test images: {100 * correct / total} %\n"
    )
