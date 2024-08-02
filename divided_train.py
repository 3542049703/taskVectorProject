import torch
from medmnist import BreastMNIST, DermaMNIST
from torch.utils.data import DataLoader, Dataset
from utils import *
import torchvision.models as models
from torchvision.transforms import transforms
from params import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import convert2RGB

# 固定参数配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = num_classes_dict["DermaMNIST"]

# 预处理函数-转换为3*224*224图像
transform = transforms.Compose([
    transforms.Lambda(convert2RGB),  # 统一为RGB格式
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集，medMNIST的子数据集
train_dataset = DermaMNIST(split="train", download=True, size=224, transform=transform, root="./data/")
test_dataset = DermaMNIST(split="test", download=True, size=224, transform=transform, root="./data/")
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

splited_datasets = split_dataset(train_dataset, 5)

for i, tmp_train_dataset in enumerate(splited_datasets):
    # 记录log
    update_log(
        log_path="./logs/split_log.txt",
        content="=" * 50 + "\n" + f"Now on sub-Dataset {i + 1}\n"
    )

    # 保存数据集
    save_dataset(tmp_train_dataset, f"./datasets/dataset_{i + 1}.pkl")

    # 定义当前data loader
    tmp_train_loader = DataLoader(dataset=tmp_train_dataset, batch_size=batch_size, shuffle=True)

    model = torch.load("./checkpoints/pretrained_checkpoint.pt").to(device)

    # 定义损失函数的权重
    class_counts = torch.tensor(check_data(dataset=tmp_train_dataset, num_classes=num_classes))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0, verbose=True)

    total_steps = len(tmp_train_loader)
    for epoch in range(epochs):
        model=model.train()
        for step, (images, labels) in enumerate(tmp_train_loader):
            images = images.to(device=device)
            labels = labels.reshape(-1).to(torch.int64).to(device=device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 20 == 0:
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
                log_path="./logs/split_log.txt",
                content=f"Epoch {epoch + 1}:Accuracy of the network on the {len(test_dataset)} test images: {100 * correct / total} %\n"
            )
    # 保存参数文件
    model=model.eval()
    torch.save(model.cpu(), f"./checkpoints/fintuned_checkpoint_{i + 1}.pt")
