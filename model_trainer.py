import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import gc

def set_seed(seed=1998):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
class LightTinyCNN(nn.Module):#只用来计算flop
    def __init__(self, num_classes=2):
        super(LightTinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(32 * 4 * 4, 64)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1) 
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)

        return x
        
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    
class LOSNLOS_CNN_Adaptive(nn.Module):
    def __init__(self):
        super(LOSNLOS_CNN_Adaptive, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # 使用 AdaptiveAvgPool1d 保证输出长度固定
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=16)  # 输出为 [B, 128, 16]

        self.fc1 = nn.Linear(128 * 16, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # 输入: [B, 1, L]，如 [B, 1, 4096]
        x = F.relu(self.conv1(x))     # [B, 64, L]
        x = F.relu(self.conv2(x))     # [B, 128, L]
        x = self.pool(x)              # [B, 128, L/2]
        x = self.global_pool(x)       # [B, 128, 16]
        x = x.view(x.size(0), -1)     # [B, 128*16]
        x = F.relu(self.fc1(x))       # [B, 64]
        x = self.fc2(x)               # [B, 2]
        return x

class AttentionCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.attention = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        att = self.attention(x)
        x = x * att
        x = self.fc(x)
        return x

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(CNN_BiLSTM, self).__init__()

        # CNN Block
        self.cnn_block = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((32, 1))  # 输出为 (B, 512, 32, 1)
        )

        # Bi-LSTM Block
        self.bilstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: (B, 1, 256, 256)
        x = self.cnn_block(x)  # -> (B, 512, 32, 1)
        x = x.squeeze(-1)      # -> (B, 512, 32)
        x = x.permute(0, 2, 1) # -> (B, 32, 512)

        lstm_out, _ = self.bilstm(x)  # -> (B, 32, 512)
        out = lstm_out[:, -1, :]      # 取最后一个时间步 (B, 512)

        out = self.classifier(out)    # -> (B, num_classes)
        return out
    
def summary_model(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)
    print(f"模型参数总量：{total:,}")
    print(f"可训练参数量：{trainable:,}")
    print(f"模型大小约：{size_mb:.2f} MB")

def prepare_data(train_dir, test_dir, batch_size=16, seed=1998):
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # 设置全局 Generator（用于 random_split）
    split_generator = torch.Generator().manual_seed(seed)

    full_dataset = datasets.ImageFolder(train_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=split_generator)

    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # 设置用于 DataLoader 随机打乱的 generator
    loader_generator = torch.Generator().manual_seed(seed)

    # worker_init_fn 也要设置 seed，以确保每个 worker 初始化一致
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
        generator=loader_generator, worker_init_fn=seed_worker
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, test_loader, model_name, epochs=40, save_dir="models",lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = get_baseline_model(name=model, num_classes=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-5)  # 100个epoch

    for epoch in range(epochs):
        model.train()
        scheduler.step()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc, _, _, _ = evaluate_model(model, val_loader, device)
        test_acc, _, _, _ = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}- Test Acc: {test_acc:.4f}")
        

    # Evaluate and save
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    print(test_metrics)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved final model weights to: {save_path}")

    txt_path = os.path.join(save_dir, f"{model_name}_metrics.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Validation:\n")
        f.write(f"Accuracy: {val_metrics[0]:.4f}\n")
        f.write(f"Precision: {val_metrics[1]:.4f}\n")
        f.write(f"Recall: {val_metrics[2]:.4f}\n")
        f.write(f"F1-score: {val_metrics[3]:.4f}\n\n")
        f.write(f"Test:\n")
        f.write(f"Accuracy: {test_metrics[0]:.4f}\n")
        f.write(f"Precision: {test_metrics[1]:.4f}\n")
        f.write(f"Recall: {test_metrics[2]:.4f}\n")
        f.write(f"F1-score: {test_metrics[3]:.4f}\n")
    print(f"Saved metrics to: {txt_path}")
    summary_model(model)
    del model
    del optimizer
    del scheduler
    del criterion
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def evaluate_model(model, loader, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            _, predicted = torch.max(preds, 1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, precision, recall, f1

def get_baseline_model(name='resnet18', num_classes=2):
    if name == 'simplecnn':
        return SimpleCNN(num_classes=num_classes)
    elif name == 'TinyCNN':
        model = LightTinyCNN(num_classes=num_classes)
        return model
    elif name == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif name == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=False)
    elif name == 'vgg16':
        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(1, 64, 3, padding=1)
        model.classifier[-1] = nn.Linear(4096, num_classes)
        return model
    elif name == 'densenet121':
        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    elif name == 'attentioncnn':
        return AttentionCNN(num_classes=num_classes)
    elif name == 'bilstm':
        return BiLSTMImageClassifier(num_classes=num_classes)
    elif name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(pretrained=False)
        model.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1, bias=False)  # 改输入通道为1
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)  # 改输入通道为1
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    elif name == 'CNN_BiLSTM':
        model = CNN_BiLSTM(input_channels=1, num_classes=num_classes)
        return model
    else:
        raise ValueError(f"error: {name}")
    

    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model