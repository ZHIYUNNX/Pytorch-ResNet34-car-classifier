# %%
from torchvision import models
from dataset import Car_Dataset
from model import resnet34_stanfordcars
from glob import glob
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import time
import sys

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

print(torch.cuda.is_available())
print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# %%

root_dir = r'E:\Learning_materials\AI_Project\AI_Course\Task\Car_Project_2\data'
train_root = os.path.join(root_dir, 'train')
test_root  = os.path.join(root_dir, 'test')
input_root = os.path.join(root_dir, 'input')


train_df = pd.read_csv(os.path.join(root_dir,'anno_train.csv'), header=None)
test_df = pd.read_csv(os.path.join(root_dir, 'anno_test.csv'), header=None)

train_df.columns = ['image_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id']
test_df.columns = ['image_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'class_id']

all_train_paths = [os.path.join(train_root, fn) for fn in train_df['image_name']]
all_train_labels = (train_df['class_id'].to_numpy() - 1)
all_train_bboxes = train_df[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()
print(all_train_labels.max(), all_train_labels.min())

test_paths = [os.path.join(test_root, fn) for fn in test_df['image_name']]
test_labels = (test_df['class_id'].to_numpy() - 1)
test_bboxes = test_df[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].to_numpy()

train_paths, val_paths, train_labels, val_labels, train_bboxes, val_bboxes = train_test_split(
    all_train_paths,
    all_train_labels,
    all_train_bboxes,
    test_size=0.1,
    stratify=all_train_labels,
    random_state=42
)
# print(train_labels)
# print(val_labels)



# %%


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.AutoAugment(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]) 

train_set = Car_Dataset(train_paths, train_labels, train_bboxes, transform=train_transform,Crop = True)
val_set   = Car_Dataset(val_paths, val_labels, val_bboxes, transform=val_transform, Crop = True)
test_set = Car_Dataset(test_paths, test_labels, test_bboxes, transform=val_transform, Crop = True)



train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

#%%
logwriter = SummaryWriter('runs/car_dataset_experiment_3')

# step = 0
# for data in train_loader:
#     images, labels = data
#     logwriter.add_images('train_images', images, step)
#     step += 1
#     if step == 100:  # Log only the first 10 batches
#         break
# logwriter.close()


# %%
# net = models.resnet34(pretrained=True)
# net.fc = nn.Linear(in_features=net.fc.in_features, out_features=196)
net = resnet34_stanfordcars()
net.to(device)
epochs = 20
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(params = net.parameters(), lr=0.001,  weight_decay=1e-4) 
lrscheduler = OneCycleLR(optimizer, 0.001, epochs=epochs, steps_per_epoch=len(train_loader))

load_path = r'./weights/ResNet34_pretrained.pth'  # 设置预训练权重路径
save_path = r'./weights/resnet34_stanfordcars_2.pth' # 设置存储权重路径
if os.path.exists(load_path):
    net.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
else:
    print("未找到权重文件，使用随机初始化参数")


best_acc = 0.5  # 初始最佳准确率
train_num = len(train_set)
val_num   = len(val_set)
test_num  = len(test_set)
train_losses = []
train_accuracies = []
val_accuracies = []

total_start_time = time.time() 
print("Training started...")
for epoch in range(epochs):
    # train
    epoch_start_time = time.time()
    print("Epoch: {}".format(epoch+1))
    net.train()  
    running_loss = 0.0  # 用来累加训练中的损失
    running_correct = 0.0
    loss_history = []

    for step, data in enumerate(train_loader, start=0):
        #   获取数据的图像和标签
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        #   将历史损失梯度清零
        optimizer.zero_grad()

        #   参数更新
        outputs = net(images)                   
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)       
        loss.backward()                                    
        optimizer.step()                                   

        lrscheduler.step()
        #   打印统计信息
        running_loss += loss.item()
        running_correct += (labels == predicted).sum().item()
        loss_history.append(loss.item())
        #   打印训练进度
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f},correct:{}".format(int(rate * 100), a, b, loss,running_correct), end="")
    print()

    train_loss = running_loss / len(train_loader)
    train_acc = running_correct / train_num
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    logwriter.add_scalar('Training Loss', train_loss, epoch)
    logwriter.add_scalar('Training Accuracy', train_acc, epoch)
    #validate
    net.eval()  
    acc = 0.0
    with torch.no_grad():
        for data_val in val_loader:
            val_images, val_labels = data_val
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        accurate_val = acc / val_num
        val_accuracies.append(accurate_val)

        logwriter.add_scalar('Validation Accuracy', accurate_val, epoch)

        if accurate_val > best_acc:
            best_acc = accurate_val
            #   保存当前权重
            torch.save(net.state_dict(), save_path)
            
        torch.save(net.state_dict(), r'./weights/last_epoch111.pth')
        epoch_end_time = time.time()  # 记录本轮结束时间
        epoch_elapse = epoch_end_time - epoch_start_time
        total_elapse = epoch_end_time - total_start_time
        print("[epoch %d] train_loss: %.5f  test_accuracy: %.5f  train_accuracy: %.5f  epoch_elapse: %.2fs  total_elapse: %.2fs"%
              (epoch + 1, train_loss, accurate_val, train_acc, epoch_elapse, total_elapse))

        logwriter.add_scalar('Test Accuracy', accurate_val, epoch)  
 
logwriter.close()


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.legend()

plt.tight_layout()
plt.show()

print("Finished Training")

