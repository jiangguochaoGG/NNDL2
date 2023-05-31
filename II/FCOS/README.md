# FCOS在VOC数据集上的目标检测

项目文件架构：

```{plain}
FCOS
├── model_data          # VOC数据集的参数文件夹
├── nets                # 网络结构的定义
├── utils               # 网络结构的定义
├── VOCdevkit
│   └── VOC2007         # VOC数据集的格式
│       ├── Annotations
│       ├── ImageSets
│       └── JPEGImages
├── fcos.py             # 具体FCOS网络的一些功能
├── get_map.py          # 计算mAP
├── predict.py          # 预测
├── summary.py          # 总结
├── train.py            # 训练
├── voc_annotation.py   # 标注数据集
└── README.md           # 该文件
```

## 数据集准备

1. 下载VOC数据集，这里提供一个VOC0712的网盘下载链接：<https://pan.baidu.com/s/1AYao-vYtHbTRN-gQajfHCw>，密码7yyp
2. 将数据集保存到`datasets`文件夹中，更名为`VOC2007`，保存为上述文件架构形式
3. 修改`voc_annotation.py`里面的`annotation_mode=2`，运行`voc_annotation.py`生成根目录下的`2007_train.txt`和`2007_val.txt`

## 训练

`python train.py`即可开始训练

## 预测

训练结果预测需要用到两个文件，分别是`fcos.py`和`predict.py`。我们首先需要去`fcos.py`里面修改`model_path`以及`classes_path`，这两个参数必须要修改。完成修改后就可以运行`predict.py`进行检测，运行后输出图片路径即可检测
