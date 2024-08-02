import torch
from medmnist import DermaMNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torchvision.models as models
from params import *
from taskVetcor import TaskVector
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

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model = change_classification_heads(model, in_features=512, new_num_classes=num_classes)

taskVec_1 = TaskVector("./checkpoints/pretrained_checkpoint.pt", "./checkpoints/fintuned_checkpoint_1.pt")
# taskVec = taskVec_1
taskVec_2 = TaskVector("./checkpoints/pretrained_checkpoint.pt", "./checkpoints/fintuned_checkpoint_2.pt")
# taskVec = taskVec_2
taskVec_3 = TaskVector("./checkpoints/pretrained_checkpoint.pt", "./checkpoints/fintuned_checkpoint_3.pt")
# taskVec = taskVec + taskVec_3
taskVec_4 = TaskVector("./checkpoints/pretrained_checkpoint.pt", "./checkpoints/fintuned_checkpoint_4.pt")
# taskVec=taskVec+taskVec_4
taskVec_5 = TaskVector("./checkpoints/pretrained_checkpoint.pt", "./checkpoints/fintuned_checkpoint_5.pt")
# taskVec=taskVec+taskVec_5

taskVec = taskVec_3

model = taskVec.apply_to("./checkpoints/pretrained_checkpoint.pt").to(device)

model = model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device=device)
        labels = labels.reshape(-1).to(torch.int64).to(device=device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    update_log(
        log_path="./logs/merge_log.txt",
        content=f"task vector model :Accuracy of the network on the {len(test_dataset)} test images: {100 * correct / total} %\n"
    )
