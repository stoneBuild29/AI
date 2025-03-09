import numpy as np

# 定义目标函数 f(x) = (x - 3)^2
def f(x):
    return (x - 3)**2

# 定义目标函数的梯度
def grad_f(x):
    return 2 * (x - 3)

# 随机初始化 x 的值
x = np.random.randn()
learning_rate = 0.1  # 学习率
num_steps = 50       # 迭代步数

print("Initial x:", x, "f(x):", f(x))
for i in range(num_steps):
    # 模拟随机梯度：在真实梯度上加上少量噪声
    grad = grad_f(x) + np.random.randn() * 0.1
    x = x - learning_rate * grad
    if i % 10 == 0:
        print(f"Step {i}: x = {x:.4f}, f(x) = {f(x):.4f}")
print("Final x:", x, "f(x):", f(x))
