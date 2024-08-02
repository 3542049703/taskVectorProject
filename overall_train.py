import torch
from medmnist import BreastMNIST, DermaMNIST
from torch.utils.data import DataLoader, Dataset
from utils import *
import torchvision.models as models
from torchvision.transforms import transforms
from params import *
from torch.optim.lr_scheduler import CosineAnnealingLR

# 固定参数配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = num_classes_dict["DermaMNIST"]

# 加载预训练模型
# 输入为(batch_size,3,224,224)
# 输出为(1,1000)，表示1000个类别的概率分布
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# model = models.resnet18(weights=None).to(device)
torch.save(model.cpu(),"./checkpoints/pretrained_checkpoint.pt")

# 预处理函数-转换为3*224*224图像
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

# 改变分类头，默认初始化方式
model = change_classification_heads(model, in_features=512, new_num_classes=num_classes).to(device)

# 定义损失函数的权重
class_counts = torch.tensor(check_data(dataset=train_dataset, num_classes=num_classes))
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, )
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, verbose=True)

total_steps = len(train_loader)
for epoch in range(epochs):
    model=model.train()
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device=device)
        labels = labels.reshape(-1).to(torch.int64).to(device=device)
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 23 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{step + 1}/{total_steps}], Loss: {loss.item()}")
    scheduler.step()

    model=model.eval()
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
            log_path="./logs/log.txt",
            content=f"Epoch {epoch + 1}:Accuracy of the network on the {len(test_dataset)} test images: {100 * correct / total} %\n"
        )
