# %%
from torchvision import models
from glob import glob
from dataset import Car_Dataset
from model import resnet34_stanfordcars
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))


print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

epochs = 1

# %%

root_dir = r'E:\Learning_materials\AI_Project\AI_Course\Task\Car_Project_2\data'

test_root  = os.path.join(root_dir, 'test')



test_df = pd.read_csv(os.path.join(root_dir, 'anno_test.csv'), header=None)
classname_df = pd.read_csv(os.path.join(root_dir, 'names.csv'), header=None)

print(classname_df[0][1])

test_df.columns = ['image_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id']


test_paths = [os.path.join(test_root, fn) for fn in test_df['image_name']]
test_labels = (test_df['class_id'].to_numpy() - 1)
test_bboxes = test_df[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
# print(test_labels.max(), test_labels.min())


# %%

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

test_set   = Car_Dataset(test_paths, test_labels, test_bboxes, transform=val_transform, Crop = True)
test_loader   = DataLoader(test_set, batch_size=32, shuffle=False)


test_num   = len(test_set)


# %%
net = resnet34_stanfordcars()
net.fc = torch.nn.Linear(net.fc.in_features, 196)
# net = models.vgg11_bn(pretrained=False)
# net.classifier[6] = nn.Linear(in_features=net.classifier[6].in_features, out_features=196)
net.to(device)

#   设置存储权重路径
save_path = r'./weights/resnet34_stanfordcars_2.pth'
net.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))

num_classes = 196
class_correct = np.zeros(num_classes)
class_total = np.zeros(num_classes)

net.eval()  
acc = 0.0
with torch.no_grad():
    for step, data_test in enumerate(test_loader, start = 0):
        test_images, test_labels = data_test
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        outputs = net(test_images)
        predict_y = torch.max(outputs, dim=1)[1]
        acc += (predict_y == test_labels).sum().item()
        rate = (step + 1) / len(test_loader)
        for i in range(len(test_labels)):
            label = test_labels[i]
            pred = predict_y[i]
            class_total[label] += 1
            if label == pred:
                class_correct[label] += 1
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\r {:^3.0f}%[{}->{}]{}".format(int(rate * 100), a, b, acc), end="")
        
    accurate_val = acc / test_num
    print()
    print("test_accuracy: %.5f  "%
              (accurate_val))   
    
class_acc = class_correct / class_total
print("Test Finished")

for i, acc in enumerate(class_acc):
    print(f"The accuracy of class {(i+1)} : {acc:.2%}")

fig, axs = plt.subplots(2, 1, figsize=(16, 12))  

# 第一张图：准确率分布
axs[1].bar(range(num_classes), class_acc)
axs[1].set_xlabel('classes')
axs[1].set_ylabel('accuracy_rate')
axs[1].set_title('The distribution of prediction accuracy rates for each category')
axs[1].set_xticks(range(1, num_classes, max(1, num_classes // 20)))
axs[1].set_ylim(0, 1)

# 第二张图：样本数分布
axs[0].bar(range(num_classes), class_total)
axs[0].set_xlabel('classes')
axs[0].set_ylabel('total_count')
axs[0].set_title('The distribution of total counts for each category')
axs[0].set_xticks(range(1, num_classes, max(1, num_classes // 20)))
axs[0].set_ylim(0, int(max(class_total)))
axs[0].set_yticks(range(0, int(max(class_total)), 10))

plt.tight_layout()
plt.show()

# single test

# images_root = './inputs'
# images_paths = glob(os.path.join(images_root, '*.jpg'))
# net.eval()
# with torch.no_grad():
#     for image_path in images_paths:
#         image = Image.open(image_path).convert('RGB')
#         # 预处理图片
#         input_image = val_transform(image).unsqueeze(0).to(device)
#         # 模型预测
#         output = net(input_image)
#         predict_y = torch.max(output, dim=1)[1].item()
#         class_id = predict_y + 1  
#         class_name = classname_df[0][predict_y]

#         # 显示图片和预测结果
#         plt.imshow(image)
#         plt.title(f"Class ID: {class_id}, Class Name: {class_name}")
#         plt.axis('off')
#         plt.show()