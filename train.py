import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast  # 导入混合精度训练
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU，如果没有则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据预处理和加载
# 使用 torchvision.transforms 对图像进行一系列的预处理操作
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # 调整输入图像的大小为 227x227 (符合 AlexNet 的要求)
    transforms.ToTensor(),  # 将图像转换为 Tensor 格式
    transforms.Normalize((0.5,), (0.5,))  # 标准化操作，均值0.5，标准差0.5
])

# 下载并加载 MNIST 数据集，数据集已经被预处理
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 使用 DataLoader 加载训练集和测试集，设置 batch size 和多线程加载
trainloader = DataLoader(trainset, batch_size=128, num_workers=2, pin_memory=True)
testloader = DataLoader(testset, batch_size=128, num_workers=2, pin_memory=True)

# 2. 定义 AlexNet 网络模型
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 定义卷积层和池化层
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 第一个卷积层
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化层
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 第二个卷积层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 第三个卷积层
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 第四个卷积层
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 第五个卷积层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 池化层
        )
        # 定义全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Dropout 层，防止过拟合
            nn.Linear(256 * 6 * 6, 4096),  # 第一个全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 第二个全连接层
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)  # 输出层，10 个类别
        )

    def forward(self, x):
        # 前向传播
        x = self.features(x)  # 卷积层
        x = x.view(x.size(0), -1)  # 展平数据
        x = self.classifier(x)  # 全连接层
        return x


# 3. 创建模型实例并将其移动到 GPU 上
model = AlexNet().to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类问题常用的损失函数
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # 使用 AdamW 优化器

# 用于保存训练过程中的损失和准确率
train_losses = []
train_accuracies = []

# 5. 训练模型
def train_model():
    epochs = 5  # 训练周期数
    accumulation_steps = 4  # 梯度累积的步骤数（当前未使用）
    scaler = GradScaler()  # 初始化混合精度训练的 GradScaler
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0  # 初始化当前 epoch 的损失
        correct = 0  # 记录正确的预测个数
        total = 0  # 记录总的样本数
        print(f"Epoch [{epoch + 1}/{epochs}] started.")

        # 使用 tqdm 包装 trainloader 以显示进度条
        for i, (inputs, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100), 1):
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据和标签移动到 GPU 上

            optimizer.zero_grad()  # 清空优化器中的梯度信息

            with autocast():  # 启用混合精度训练
                outputs = model(inputs)  # 获取模型输出
                loss = criterion(outputs, labels)  # 计算损失

            scaler.scale(loss).backward()  # 反向传播计算梯度
            if (i + 1) % accumulation_steps == 0:  # 每 accumulation_steps 次更新一次梯度（目前无效）
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item()  # 累加当前 batch 的损失
            _, predicted = torch.max(outputs, 1)  # 获取模型的预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确的预测个数

        # 计算本轮训练的平均损失和准确率
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = correct / total * 100
        train_losses.append(epoch_loss)  # 保存当前 epoch 的损失
        train_accuracies.append(epoch_accuracy)  # 保存当前 epoch 的准确率
        print(f"Epoch [{epoch + 1}/{epochs}] finished. Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%\n")

    # 6. 保存模型
    torch.save(model.state_dict(), 'alexnet_mnist.pth')  # 保存模型权重
    print("Model saved successfully!")

# 7. 绘制训练损失和准确率的图
def plot_training_progress():
    plt.figure(figsize=(12, 6))  # 创建一个宽12英寸、高6英寸的图形窗口

    # 绘制训练损失的子图
    plt.subplot(1, 2, 1)  # 1行2列的第一个子图
    plt.plot(range(1, 6), train_losses, marker='o', label='Train Loss')
    plt.title('Train Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    # 绘制训练准确率的子图
    plt.subplot(1, 2, 2)  # 1行2列的第二个子图
    plt.plot(range(1, 6), train_accuracies, marker='o', label='Train Accuracy')
    plt.title('Train Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    # 调整布局，避免子图重叠
    plt.tight_layout()

    # 显示图形窗口
    plt.show()

# 8. 运行代码
if __name__ == '__main__':
    # 设置 multiprocessing 的启动方法为 'spawn'（Windows 需要）
    multiprocessing.set_start_method('spawn')

    # 开始训练模型
    train_model()

    # 绘制训练过程图
    plot_training_progress()
