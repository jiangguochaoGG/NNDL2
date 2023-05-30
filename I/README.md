# 神经网络和深度学习 第二次作业 Part I

## 项目架构

```{plain}
I
├── data         # 保存CIFAR-100数据集
├── model.py     # 模型ResNet的实现
├── utils.py     # 数据增强方法的实现
├── visualize.py # 可视化图片的实现
├── dataset.py   # 数据集的实现
└── main.py      # 主函数，训练
```

直接执行`python main.py`即可训练模型，具体的参数可以进入`main.py`中详细查看和修改。在训练过程中，会开启一个tensorboard以便查看训练统计数据，包括训练损失、验证损失和精度变化，在训练完成后会选择训练中在验证精度上表现最好的模型在测试集上进行测试，得到最后的测试精度。