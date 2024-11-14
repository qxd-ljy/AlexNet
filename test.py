import torch
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader

# 检查是否有可用的 GPU，如果没有则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 数据预处理和加载
# 定义对图像的转换操作：调整大小、转换为Tensor、标准化
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # 调整输入图像的大小为 227x227 (符合 AlexNet 的要求)
    transforms.ToTensor(),  # 将图像转换为 Tensor 格式
    transforms.Normalize((0.5,), (0.5,))  # 标准化操作，均值0.5，标准差0.5
])

# 加载 MNIST 数据集（训练集），并应用定义的图像转换
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False)  # 使用 DataLoader 批量加载数据

# 2. 定义 AlexNet 模型结构
# 这个模型是基于经典的 AlexNet 结构，只不过输入是灰度图像（1通道），而非 RGB 图像（3通道）
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 特征提取部分（卷积层 + 激活函数 + 池化层）
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # 输入1通道（灰度图），输出64通道
            nn.ReLU(inplace=True),  # 激活函数 ReLU
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化层
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 第二个卷积层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 池化层
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 第三个卷积层
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 第四个卷积层
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 第五个卷积层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 池化层
        )
        # 分类器部分（全连接层）
        self.classifier = nn.Sequential(
            nn.Dropout(),  # Dropout 层，防止过拟合
            nn.Linear(256 * 6 * 6, 4096),  # 全连接层1，输入大小 256*6*6，输出 4096
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # 全连接层2，输出 4096
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)  # 输出层，10个类别（MNIST 0-9）
        )

    def forward(self, x):
        # 前向传播过程
        x = self.features(x)  # 通过卷积层提取特征
        x = x.view(x.size(0), -1)  # 展平数据（展平成一维）
        x = self.classifier(x)  # 通过全连接层进行分类
        return x


# 3. 加载训练好的模型
# 此函数用于加载保存的模型权重
def load_model(model_path='alexnet_mnist.pth'):
    model = AlexNet().to(device)  # 创建模型实例并移动到设备（GPU/CPU）
    model.load_state_dict(torch.load(model_path))  # 加载模型权重
    model.eval()  # 设置模型为评估模式
    print("Model loaded successfully!")
    return model


# 4. 批量进行图片预测
# 该函数从 dataloader 中获取指定数量的图像，并进行预测
def batch_predict_images(model, dataloader, num_images=6):
    predictions = []  # 用于保存预测结果
    images = []  # 用于保存输入图像
    labels = []  # 用于保存实际标签

    # 不计算梯度以提高效率
    with torch.no_grad():
        for i, (input_images, input_labels) in enumerate(dataloader):
            if i * 64 >= num_images:  # 控制处理的图像数量
                break

            input_images = input_images.to(device)  # 将图像数据转移到 GPU 上
            input_labels = input_labels.to(device)  # 将标签数据转移到 GPU 上

            # 通过模型进行预测
            outputs = model(input_images)
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别

            predictions.extend(predicted.cpu().numpy())  # 保存预测结果到 CPU 上
            images.extend(input_images.cpu().numpy())  # 保存输入图像到 CPU 上
            labels.extend(input_labels.cpu().numpy())  # 保存真实标签到 CPU 上

    return images[:num_images], labels[:num_images], predictions[:num_images]


# 5. 可视化批量图像及其预测
# 该函数显示图像和预测结果
def visualize_images(images, labels, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))  # 创建 2x3 的图像子图
    axes = axes.ravel()  # 将子图展平成一维数组

    for i in range(6):
        image = images[i].squeeze()  # 去掉多余的维度（例如[1, 227, 227] -> [227, 227]）
        ax = axes[i]
        ax.imshow(image, cmap='gray')  # 显示图像，使用灰度图
        ax.set_title(f"Pred: {predictions[i]} | Actual: {labels[i]}")  # 显示预测标签和实际标签
        ax.axis('off')  # 关闭坐标轴显示

    plt.tight_layout()  # 调整子图之间的间距
    plt.show()  # 显示图像


# 6. 主函数
if __name__ == '__main__':
    # 设置 multiprocessing 的启动方法为 'spawn'，用于兼容不同操作系统（Windows需要）
    multiprocessing.set_start_method('spawn')

    # 加载训练好的模型
    model = load_model()

    # 获取前6张图像及其预测结果
    images, labels, predictions = batch_predict_images(model, trainloader, num_images=6)

    # 可视化这些图像及其预测结果
    visualize_images(images, labels, predictions)
