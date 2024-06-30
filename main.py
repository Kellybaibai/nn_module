# -*- coding: utf-8 -*-
from src.data_process import Dataset, CustomDataLoader
from src.Residual_MLP_model import ResidualMLP
import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm

torch.manual_seed(123)

def train(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(torch.float32)  # 将输入数据转换为 Float 类型
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# 测试函数
def predict(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float32)  # 将输入数据转换为 Float 类型
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy * 100

# Training function
def train_mnist(model, train_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = inputs.view(-1, 28*28)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

def predict_mnist(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs = inputs.view(-1, 28*28)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


if __name__ == '__main__':

    # 创建数据集
    iris_dataset = Dataset(dataset_name='iris')
    mnist_dataset = Dataset(dataset_name='mnist', root='./data')
    fashion_mnist_dataset = Dataset(dataset_name='fashion-mnist', root='./data')
    iris_loader = CustomDataLoader(dataset=iris_dataset, batch_size=32)
    iris_train_loader, iris_test_loader = iris_loader.split_train_test(test_size=0.2)
    mnist_loader = CustomDataLoader(dataset=mnist_dataset, batch_size=64)
    mnist_train_loader, mnist_test_loader = mnist_loader.split_train_test(test_size=0.2)
    fashion_mnist_loader = CustomDataLoader(dataset=fashion_mnist_dataset, batch_size=32)
    fashion_mnist_train_loader, fashion_mnist_test_loader = fashion_mnist_loader.split_train_test(test_size=0.2)

    # 定义模型、损失函数和优化器
    model = ResidualMLP(input_dim=4, hidden_dim=64, output_dim=3)  # 根据具体数据集调整输入维度和输出维度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 在 Iris 数据集上训练和测试
    train(model, iris_train_loader, criterion, optimizer, num_epochs=10)
    accuracy = predict(model, iris_test_loader)
    print(f'Iris Dataset Accuracy: {accuracy:.2f}%')

    # 定义模型、损失函数和优化器
    model = ResidualMLP(input_dim=28*28, hidden_dim=128, output_dim=10)  # 根据具体数据集调整输入维度和输出维度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 在 MNIST 数据集上训练和测试
    train_mnist(model, mnist_train_loader, criterion, optimizer, num_epochs=15)
    accuracy = predict_mnist(model, mnist_test_loader)
    print(f'MNIST Dataset Accuracy: {accuracy:.2f}%')

    # 定义模型、损失函数和优化器
    model = ResidualMLP(input_dim=28*28, hidden_dim=28, output_dim=10)  # 根据具体数据集调整输入维度和输出维度
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    # 在 Fashion-MNIST 数据集上训练和测试
    train_mnist(model, fashion_mnist_train_loader, criterion, optimizer, num_epochs=20)
    accuracy = predict_mnist(model, fashion_mnist_test_loader)
    print(f'Fashion-MNIST Dataset Accuracy: {accuracy:.2f}%')
