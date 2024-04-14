import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, vit_l_16
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import csv
import numpy as np
#vit版本，两台4090跑

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureExtractor, self).__init__()
        self.model = vit_l_16(pretrained=pretrained)
        self.pretrained = pretrained 

    def forward(self, x):
        if self.pretrained:
            x = self.model(x)
        else:
            self.model.head = nn.Identity()  
            self.out_features = 1000 
            x = self.model(x)
        return x
    def remove_last_layer(self):
        self.pretrained = False
        
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SETransformerFusion(nn.Module):
    def __init__(self, image_feat_dim, blood_feat_dim, hidden_dim, num_classes, num_layers, num_heads):
        super(SETransformerFusion, self).__init__()
        self.image_embedding = nn.Linear(image_feat_dim, hidden_dim)
        self.blood_embedding = nn.Linear(blood_feat_dim, hidden_dim)
        self.image_se = SEBlock(hidden_dim)
        self.blood_se = SEBlock(hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, image_feat, blood_feat):
        image_embed = self.image_embedding(image_feat)
        blood_embed = self.blood_embedding(blood_feat)
        image_embed = self.image_se(image_embed.unsqueeze(-1)).squeeze(-1)
        blood_embed = self.blood_se(blood_embed.unsqueeze(-1)).squeeze(-1)
        x = torch.stack([image_embed, blood_embed], dim=1)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x
    
class MultimodalDataset(Dataset):
    def __init__(self, image_dataset, blood_data):
        self.image_dataset = image_dataset
        self.blood_data = blood_data
        
    def __len__(self):
        return len(self.image_dataset)
    
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        blood_feat = torch.tensor(self.blood_data[idx], dtype=torch.float32)
        return image, blood_feat, label
    
class Trainer:
    def __init__(self, feature_extractor, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = feature_extractor.to(self.device)
        self.config = config

    def pretrain(self, loader):
        optimizer = optim.SGD(self.feature_extractor.parameters(), lr=self.config["lr"], momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        self.feature_extractor.train()
        for epoch in range(self.config["pretrain_epochs"]):
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.feature_extractor(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(f"Pretrain Epoch [{epoch+1}/{self.config['pretrain_epochs']}], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}")

# 定义图像预处理
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载自定义数据集
data_path = 'data/covid'  # 数据集路径
train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
test_dataset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)

num_classes = len(train_dataset.classes)
blood_data_dim = 10

# 为每个类别生成随机血常规数据并保存到CSV文件
blood_data_ranges = {i: (i, i+1) for i in range(num_classes)}
train_blood_data = []
test_blood_data = []

for idx, (_, label) in enumerate(train_dataset):
    blood_feat = torch.FloatTensor(blood_data_dim).uniform_(*blood_data_ranges[label])
    train_blood_data.append(blood_feat.numpy())

for idx, (_, label) in enumerate(test_dataset):
    blood_feat = torch.FloatTensor(blood_data_dim).uniform_(*blood_data_ranges[label])
    test_blood_data.append(blood_feat.numpy())

with open('train_blood_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_blood_data)

with open('test_blood_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(test_blood_data)


# 从CSV文件读取血常规数据
with open('train_blood_data.csv', 'r') as file:
    reader = csv.reader(file)
    train_blood_data = [[float(x) for x in row] for row in reader]

with open('test_blood_data.csv', 'r') as file:
    reader = csv.reader(file)
    test_blood_data = [[float(x) for x in row] for row in reader]

# 创建多模态数据集
train_multimodal_dataset = MultimodalDataset(train_dataset, train_blood_data)  
test_multimodal_dataset = MultimodalDataset(test_dataset, test_blood_data)

# 加载数据
train_dataloader = DataLoader(train_multimodal_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_multimodal_dataset, batch_size=32, shuffle=False)

# 提取图像特征
feature_extractor = FeatureExtractor(pretrained=True)
feature_extractor = FeatureExtractor().to(device)
feature_extractor = nn.DataParallel(feature_extractor)

# 训练主干网络(特征提取器)
criterion = nn.CrossEntropyLoss() 
optimizer_backbone = optim.Adam(feature_extractor.parameters(), lr=0.001, weight_decay=0.01)
num_epochs_backbone = 1

for epoch in range(num_epochs_backbone):
    feature_extractor.train()
    running_loss = 0.0
    
    for images, _, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer_backbone.zero_grad()
        outputs = feature_extractor(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_backbone.step()

        running_loss += loss.item()

    print(f"Backbone Epoch [{epoch+1}/{num_epochs_backbone}], Loss: {running_loss/len(train_dataloader):.4f}")

# 保存主干网络权重  
torch.save(feature_extractor.state_dict(), 'backbone_weights.pt')

# 初始化 SETransformerFusion 模型
hidden_dim = 1000
num_layers = 2
num_heads = 8
se_transformer_fusion = SETransformerFusion(1000, blood_data_dim, hidden_dim, num_classes, num_layers, num_heads)

# 将模型移动到GPU上
se_transformer_fusion = nn.DataParallel(se_transformer_fusion)
se_transformer_fusion.to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(se_transformer_fusion.parameters()), lr=0.001, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

num_epochs = 1
for epoch in range(num_epochs):
    feature_extractor.eval()
    se_transformer_fusion.train()
    
    running_loss = 0.0
    for images, blood_feat, labels in train_dataloader:
        images = images.to(device)
        blood_feat = blood_feat.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        image_feat = feature_extractor(images)
        outputs = se_transformer_fusion(image_feat, blood_feat)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}")
    
# 评估模型性能
feature_extractor.eval()
se_transformer_fusion.eval()

with torch.no_grad():
    train_image_features = torch.cat([feature_extractor(images.to(device)) for images, _ in DataLoader(train_dataset, batch_size=32)])
    train_outputs = se_transformer_fusion(train_image_features, torch.tensor(train_blood_data, dtype=torch.float32).to(device))
    _, train_preds = torch.max(train_outputs, 1)
    train_acc = accuracy_score(train_dataset.targets, train_preds.cpu())
    train_cm = confusion_matrix(train_dataset.targets, train_preds.cpu())

    test_image_features = torch.cat([feature_extractor(images.to(device)) for images, _ in DataLoader(test_dataset, batch_size=32)])
    test_outputs = se_transformer_fusion(test_image_features, torch.tensor(test_blood_data, dtype=torch.float32).to(device))
    _, test_preds = torch.max(test_outputs, 1)
    test_acc = accuracy_score(test_dataset.targets, test_preds.cpu())
    test_cm = confusion_matrix(test_dataset.targets, test_preds.cpu())

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Training Confusion Matrix:\n{train_cm}")
print(f"Testing Accuracy: {test_acc:.4f}") 
print(f"Testing Confusion Matrix:\n{test_cm}")

# 初始化新类别的数量
num_new_classes = 0

random_idx = np.random.randint(len(test_dataset))
random_image, random_label = test_dataset[random_idx]
random_image = random_image.unsqueeze(0).to(device)

# 生成随机的血常规数据（不合理）
random_blood_data = torch.FloatTensor(1, blood_data_dim).uniform_(0, 100).to(device)

# 输入模型进行分类
with torch.no_grad():
    random_image_feature = feature_extractor(random_image)
    random_output = se_transformer_fusion(random_image_feature, random_blood_data)
    _, random_pred = torch.max(random_output, 1)
    random_confidence = torch.softmax(random_output, dim=1).max().item()

#print(f"Random image true label: {random_label}")
print(f"Random image predicted label: {random_pred.item()}")
print(f"Confidence: {random_confidence:.4f}")

# 判断是否需要进行持续学习
confidence_threshold = 0.95
if random_confidence < confidence_threshold:
    print("Confidence is below threshold. Starting continual learning...")

    # 将随机图像标记为新类别
    new_class_label = num_classes + num_new_classes
    num_new_classes += 1

    # 将随机图像及其血常规数据添加到持续学习数据集中
    continual_dataset = MultimodalDataset(train_dataset + [(random_image.squeeze(0).cpu(), new_class_label)],
                                      train_blood_data + [random_blood_data.squeeze(0).cpu().numpy().tolist()])
    continual_dataloader = DataLoader(continual_dataset, batch_size=32, shuffle=True)

    # 更新输出层以适应新的类别数量
    se_transformer_fusion.module.fc = nn.Linear(hidden_dim, num_classes + num_new_classes).to(device)

    # 定义持续学习的优化器和损失函数
    continual_optimizer = optim.Adam(list(feature_extractor.parameters()) + list(se_transformer_fusion.parameters()), lr=0.001, weight_decay=0.01)
    continual_criterion = nn.CrossEntropyLoss()

    # 进行持续学习
    num_epochs_continual = 1
    for epoch in range(num_epochs_continual):
        feature_extractor.eval()
        se_transformer_fusion.train()

        running_loss = 0.0
        for images, blood_feat, labels in continual_dataloader:
            images = images.to(device)
            blood_feat = blood_feat.to(device)
            labels = labels.to(device)

            continual_optimizer.zero_grad()

            image_feat = feature_extractor(images)
            outputs = se_transformer_fusion(image_feat, blood_feat)
            loss = continual_criterion(outputs, labels)

            loss.backward()
            continual_optimizer.step()

            running_loss += loss.item()

        print(f"Continual Learning Epoch [{epoch+1}/{num_epochs_continual}], Loss: {running_loss/len(continual_dataloader):.4f}")

    # 更新类别数量
    num_classes += num_new_classes

    # 使用更新后的模型进行预测
    with torch.no_grad():
        random_image_feature = feature_extractor(random_image)
        random_output = se_transformer_fusion(random_image_feature, random_blood_data)
        _, adapted_pred = torch.max(random_output, 1)

    print(f"Adapted prediction: {adapted_pred.item()}")

else:
    print("Confidence is above threshold. No need for continual learning.")