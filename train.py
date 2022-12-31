import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from PIL import Image
batch_size=32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MyDataset(Dataset):
    def __init__(self, image_path: list, label_class: list, transform=None):
        self.image_path = image_path
        self.label_class = label_class
        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        img = Image.open(self.image_path[item]).convert('RGB')
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode".format(self.image_path[item]))
        label = self.label_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label


image_path = '../深度学习/BitmojiDataset_Sample/trainimages'
label_path = '../深度学习/BitmojiDataset_Sample/train.csv'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])
df = pd.read_csv(label_path)
images = []
labels = []
for dir in os.listdir(image_path):
    images.append(image_path + '/' + str(dir))
    if df[df.image_id == dir].iloc[0, 1] == -1:
        labels.append(0)
    else:
        labels.append(df[df.image_id == dir].iloc[0, 1])
images = np.array(images)
labels = np.array(labels)
train_image, val_image, train_label, val_label = train_test_split(images, labels, test_size=0.2, random_state=1)
train_data = MyDataset(train_image, train_label, transform)
val_data=MyDataset(val_image,val_label,transform)
train_loader = DataLoader(train_data, batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size,shuffle=False)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:    # eval model lose this layer
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:   # eval model lose this layer
            return x, aux2, aux1
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#inception结构
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)   # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)   # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

#辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


model=GoogLeNet(num_classes=2, aux_logits=True, init_weights=True)
if torch.cuda.is_available():
    model=model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)
train_steps = len(train_loader)
val_steps = len(val_loader)
val_num = len(val_data)
best_acc = 0.0
best_p = 0
best_r = 0
best_f1 = 0
best_val_loss = 1
epoch = 0
print("start training")
while epoch < 20:
    model.train()
    running_loss = 0.0
    val_loss = 0.0
    for step, data in enumerate(train_loader):
        inputs, labels = data[0], data[1]
        optimizer.zero_grad()
        train_out,train_out1,train_out2 = model(inputs)
        loss0 = criterion(train_out, labels)
        loss1 = criterion(train_out1, labels)
        loss2 = criterion(train_out2, labels)
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    model.eval()  # 验证过程中关闭 Dropout
    acc = 0.0  # accumulate accurate number / epoch
    p=0.0
    for val_step, val_data in enumerate(val_loader):
        val_images, val_labels = val_data[0], val_data[1]
        optimizer.zero_grad()
        outputs = model(val_images)
        loss3 = criterion(outputs, val_labels)
        loss3.backward()
        optimizer.step()
        val_loss += loss3.item()
        predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
        acc += torch.eq(predict_y, val_labels).sum().item()
    p = precision_score(val_labels, predict_y, average='macro')
    r = recall_score(val_labels, predict_y, average='macro')
    f1 = f1_score(val_labels, predict_y, average='macro')
    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.4f  val_loss: %.4f  val_accuracy: %.4f  P:%.4f R:%.4f  F1:%.4f' %
            (epoch + 1, running_loss / train_steps, val_loss / val_steps, val_accurate,p,r,f1))
    # 保存准确率最高的那次网络参数
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(model.state_dict(), './log.pth')
    if p>best_p:
        best_p=p
    if r>best_r:
        best_r=r
    if f1>best_f1:
        best_f1=f1
    if val_loss<best_val_loss:
        best_val_loss=val_loss
    epoch += 1
print('Finish training')
print('val_loss: %.4f   P:%.4f  R:%.4f  F1:%.4f' %
      (best_val_loss,best_p,best_r,best_f1))
