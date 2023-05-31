# Faster R-CNN在VOC数据集上的目标检测

项目文件架构：

```{plain}
Faster R-CNN
├── configs             # 模型的参数文件夹
├── dataloader          # dataloader的定义
├── network             # 网络结构的定义
├── optimizer           # 优化器的定义
├── options             # 辅助函数，包括参数读取
├── shceduler           # 调度器的定义
├── work_config         # 包含一些有效的配置文件
├── utils               # 包含一些辅助函数，包括可视化
├── datasets
│   └── voc             # VOC数据集的格式
│       ├── Annotations
│       ├── ImageSets
│       └── JPEGImages
├── train.py            # 训练
├── eval.py             # 评估
├── inference.py        # 推理
├── preview.py          # 预览
└── README.md           # 该文件
```

## 数据集准备

1. 下载VOC数据集，这里提供一个VOC0712的网盘下载链接：<https://pan.baidu.com/s/1AYao-vYtHbTRN-gQajfHCw>，密码7yyp
2. 将数据集保存到`datasets`文件夹中，更名为`voc`，保存为上述文件架构形式

## 训练

```bash
python train.py --config configs/faster_rcnn_voc.yml
```

## 验证

运行以下命令来验证模型的`mAP`指标，`--vis`可以在tensorboard上可视化结果

```bash
python eval.py --config configs/faster_rcnn_voc.yml --load 模型文件 --vis
```
