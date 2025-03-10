# 20250310-后向传播和大模型调参的区别

- complicated code的问题：为什么模型已经知道了y = 2x + 1, 所以线性回归方程在做什么？
    - 本身这不是一个真实情况，所以在模拟数据的时候在方程上增加了噪音。在现实情况下，比如房屋价格或者大模型下，数据往往不具备这样显著的特征。所以这个方程是在recover potential pattern, maybe linear and non linear.( need 正则化 and data cleaning
    - analyze the result of the example
        - the parameters: the weights and the distance （because of the distraction of the noise）
        - 拟合直线和拟合曲线/ cost function
    - where are the parameters?
        
        nn.Linear package the weight w and b 
        
        当定义 `self.linear = nn.Linear(1, 1)` 时：
        
        - **权重 `w`**：自动初始化为一个 1x1 的矩阵（即标量），对应代码中 `self.linear.weight`。
        - **偏置 `b`**：自动初始化为一个标量，对应代码中 `self.linear.bias`。
        - **前向计算**：`output = input * self.linear.weight + self.linear.bias`。
        
- 随机梯度下降和大模型微调参数的区别
    - 在概念和方法上：随机梯度下降 SGD是在每次迭代中随机选择一个样本（或者一个小批量样本）计算梯度，并更新模型参数。而在大模型调参如GPT-3、BERT等海量参数，通常在通用任务上预训练，然后在特定任务上进行调参。
    - 第一性原理：SGD是一种优化算法，通过迭代调整模型参数，使模型在真实情况效果更好。而大模型微调是通过少量领域的数据调整参数，使之更加适应特定任务。前者是考前复习，后者是考试带小抄。
    - example
        
        ```python
        #SGD
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        #自适应优化器
        AdamW
        ```
        
- 前向传播和后向传播
    - Forward Pass
        - aim: calculate the result of the model
        - process: calculate the result from the first layer to the last layer
    
    ```python
    outputs = model(x_sample)  # 调用 forward() 方法
    loss = criterion(outputs, y_sample) # 计算损失 (y_pred - y_true)^2
    
    ```
    
    - Backward Pass
        - calculate the 梯度
        - optimize the parameters
    
    ```python
    optimizer.zero_grad()  # 清空旧梯度
    loss.backward()        # 计算梯度（d(loss)/dw 和 d(loss)/db）
    optimizer.step()       # 沿梯度反方向更新参数：w = w - lr * d(loss)/dw
    ```
    
    - example
    
    ```python
    假设当前 w=1.5, b=0.8，输入 x=2，真实值 y=5（根据 y=2x+1，此处添加了噪声）：
    
    前向计算：
    y_pred = 1.5*2 + 0.8 = 3.8
    loss = (3.8 - 5)^2 = 1.44
    
    反向梯度：
    d(loss)/dw = 2*(3.8 - 5)*2 = -4.8
    d(loss)/db = 2*(3.8 - 5) = -2.4
    
    参数更新（假设 lr=0.1）：
    w = 1.5 - 0.1*(-4.8) = 1.98
    b = 0.8 - 0.1*(-2.4) = 1.04
    ```