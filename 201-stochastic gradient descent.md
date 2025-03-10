# stochastic gradient descent

1. a basic function

   - aim: to find the minimum
   - result:  near the global minimum

   ```python
   import numpy as np

   # aim function
   def f(x):
       return (x - 3)**2

   # cost function
   def grad_f(x):
       return 2 * (x - 3)

   # random initialize
   x = np.random.randn()
   learning_rate = 0.1  # 学习率
   num_steps = 50       # 迭代步数

   print("Initial x:", x, "f(x):", f(x))
   for i in range(num_steps):
       # Stochastic Gradient Descent 
       grad = grad_f(x) + np.random.randn() * 0.1
       # to find the minimum, lower right higher left, so we make original x minus a number
       x = x - learning_rate * grad
       if i % 10 == 0:
           print(f"Step {i}: x = {x:.4f}, f(x) = {f(x):.4f}")
   print("Final x:", x, "f(x):", f(x))

   ```
2. a complicated function

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   import torch
   import torch.nn as nn

   # 生成合成数据（y = 2x + 1 + 噪声）
   np.random.seed(42)
   X = np.linspace(0, 5, 100)
   y = 2 * X + 1 + np.random.normal(0, 1, 100)

   # 转换为 PyTorch 张量
   X_tensor = torch.from_numpy(X.astype(np.float32)).view(-1, 1)
   y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

   # 定义线性回归模型
   class LinearRegression(nn.Module):
       def __init__(self):
           super().__init__()
           self.linear = nn.Linear(1, 1)  # 输入输出维度均为1

       def forward(self, x):
           return self.linear(x)

   model = LinearRegression()

   # 定义损失函数和优化器（使用SGD）
   criterion = nn.MSELoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

   # 训练过程
   losses = []
   epochs = 100

   for epoch in range(epochs):
       # 随机选择一个样本（SGD的核心：每次迭代只用一个样本）
       idx = np.random.randint(0, len(X))
       x_sample = X_tensor[idx]
       y_sample = y_tensor[idx]

       # 前向传播
       outputs = model(x_sample)
       loss = criterion(outputs, y_sample)
       losses.append(loss.item())

       # 反向传播和参数更新
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       # 每10次epoch打印损失
       if (epoch + 1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

   # 可视化训练结果
   plt.figure(figsize=(12, 4))

   # 1. 原始数据与拟合直线
   plt.subplot(1, 2, 1)
   plt.scatter(X, y, label='Original Data')
   predicted = model(X_tensor).detach().numpy()
   plt.plot(X, predicted, 'r', label='Fitted Line')
   plt.xlabel('X')
   plt.ylabel('y')
   plt.legend()

   # 2. 训练损失曲线
   plt.subplot(1, 2, 2)
   plt.plot(losses)
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training Loss with SGD')

   plt.tight_layout()
   plt.show()
   ```
   ![CleanShot 2025-03-09 at 21.29.06@2x](https://cdn.statically.io/gh/stoneBuild29/MyPictures@main/upload/CleanShot%202025-03-09%20at%2021.29.06%402x.png)
