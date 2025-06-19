# Pytorch-ResNet34-car-classifier
course_task

# README

## Task Category
Image Classification

## Tags
Model of car, Stanford car dataset, ResNet34

## About the Dataset
This is the Stanford Car Dataset by classes folder. The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g., 2012 Tesla Model S or 2012 BMW M3 coupe.

- **File Format**: JPG
- **Color Space**: RGB
- **Annotations**: CSV format, including `anno_test` and `anno_train`.

For more details, please visit [Stanford Car Dataset by classes folder](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder).

## File Structure
Please ensure that the following files are in the same directory:
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`
- `data/`

To load the data correctly, you first need to download the complete data from the above dataset link, and then extract the file and adjust it to the following file structure
data/
├── img_move.py
├── train/
│   ├── img...   
│       
├── test/
│   ├── img...
│ 
├── imgs_move.py(Copy it from the main folder)      
├── names.csv
├── train_labels.csv
└── test_labels.csv

And then run imgs_move.py

you also need to modify the `root_dir` variable in `train.py` and `test.py`. Replace:
root_dir = r'E:\Learning_materials\AI_Project\AI_Course\Task\Car_Project_2\data'
with:
root_dir = r'the current path on your computer'

In the weights/ folder, there are different weights, including those for ResNet34 and VGG11_bn. resnet34_stanfordcars.pth and last_epoch.pth are the trained weights and the weights saved from the last epoch, respectively. If you want to load other weights, please modify the load_path variable in train.py and test.py to the path of your desired weight.
Testing
The result returned by the test program is the accuracy of the test set. If you want to test specific images, please place the images in the inputs/ folder(If it doesn't exist, please create one) and uncomment the "single test" section in test.py.
Running Environment
Operating System
Windows 11
Framework and Language
PyTorch 2.5.1
Python 3.9
Data Processing Libraries
Pandas 2.3.0
NumPy 2.0.2
Scikit-Learn 1.6.1
Pillow 10.4.0
Image Processing Libraries
Pillow 10.4.0
Torch Vision 0.20.1
Logging and Visualization Libraries
Matplotlib 3.9.4
TensorBoard 2.19.0

# 说明文档

## 任务类别
图像分类

## 标签
汽车模型，斯坦福汽车数据集，ResNet34

## 关于数据集
这是斯坦福汽车数据集，按类别文件夹组织。该数据集包含196个类别的16,185张汽车图片，分为8,144张训练图片和8,041张测试图片，每个类别大致按照50-50的比例分配。类别通常包括品牌、型号和年份，例如2012年特斯拉Model S或2012年宝马M3双门轿跑车。

- **文件格式**：JPG
- **色彩空间**：RGB
- **注释**：CSV格式，包括`anno_test`和`anno_train`。

更多详情，请访问 [斯坦福汽车数据集](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)。

## 文件结构
请确保以下文件在同一目录下：
- `dataset.py`
- `model.py`
- `train.py`
- `test.py`
- `data/`

为了正确加载数据，你首先需要从上面数据集链接下载完整的数据，然后将文件解压，调整成以下的文件结构
data/
├── img_move.py
├── train/
│   ├── img...   
│       
├── test/
│   ├── img...
│ 
├── imgs_move.py(将其从主文件夹中复制过来)      
├── names.csv
├── train_labels.csv
└── test_labels.csv

然后运行imgs_move.py

你还需要修改`train.py`和`test.py`中的`root_dir`变量。将：
root_dir = r'E:\Learning_materials\AI_Project\AI_Course\Task\Car_Project_2\data'
替换为：
root_dir = r'你电脑上的当前路径'
在weights/文件夹中，有不同模型的权重，包括ResNet34和VGG11_bn的权重。resnet34_stanfordcars.pth和last_epoch.pth分别是训练好的权重和最后一个训练周期保存的权重。如果你想加载其他权重，请修改train.py和test.py中的load_path变量为你想要加载的权重路径。
测试
测试程序返回的结果是测试集的准确率。如果你想测试特定的图片，请将图片放入inputs/文件夹(如果没有，请创建一个)，并取消test.py中“单张测试”部分的注释。
运行环境
操作系统
Windows 11
框架和语言
PyTorch 2.5.1
Python 3.9
数据处理库
Pandas 2.3.0
NumPy 2.0.2
Scikit-Learn 1.6.1
Pillow 10.4.0
图像处理库
Pillow 10.4.0
Torch Vision 0.20.1
日志记录和可视化库
Matplotlib 3.9.4
TensorBoard 2.19.0
