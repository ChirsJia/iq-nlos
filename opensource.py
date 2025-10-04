import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from model_trainer import train_model, prepare_data, get_baseline_model,set_seed,evaluate_model

def get_baseline_model(name='resnet18', num_classes=2):
    if name == 'shufflenet_v2':
        model = models.shufflenet_v2_x1_0(pretrained=False)
        model.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1, bias=False)  # 改输入通道为1
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"error: {name}")

class LightTinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(LightTinyCNN, self).__init__()
        self.quant = QuantStub()

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

        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = x.reshape(x.size(0), -1)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)

        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['conv3', 'bn3', 'relu3'],
                                               ['fc1', 'relu_fc']], inplace=True)
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
    cm = confusion_matrix(y_true, y_pred)   # 生成混淆矩阵
    return acc, precision, recall, f1, cm

def main():
    set_seed(1998)
    base_dir = r"png_woaug"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "CORRIDOR_test")
    for p in [train_dir, test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing directory: {p}")
    batch_size = 16
    device = torch.device("cpu")
    train_loader, val_loader, test_loader = prepare_data(train_dir, test_dir, batch_size)
    model_entries = [
        {
            "name": "TinyCNN_QAT_Script",
            "type": "script",
            "path": r"TinyCNN0.7_final_qat_quantized_jit.pt"
        },
        {
            "name": "ShuffleNetV2",
            "type": "baseline",
            "arch": "shufflenet_v2",
            "state_dict": r"shufflenet_v2_png_woaug_16_0.01new.pth"
        },
    ]
    results = []
    for entry in model_entries:
        print(f"\n=== Model: {entry['name']} ===")
        if entry["type"] == "script":
            if not os.path.isfile(entry["path"]):
                print("Missing scripted model, skip.")
                continue
            model = torch.jit.load(entry["path"], map_location=device)
            model.eval()
        elif entry["type"] == "baseline":
            model = get_baseline_model(entry["arch"], num_classes=2)
            sd_path = entry.get("state_dict")
            if sd_path and os.path.isfile(sd_path):
                try:
                    sd = torch.load(sd_path, map_location="cpu", weights_only=True)
                    model.load_state_dict(sd)
                    print("Loaded weights:", sd_path)
                except Exception as e:
                    print("Load failed, random init:", e)
            else:
                print("No weights file, using random init.")
            model.to(device).eval()
        else:
            print("Unknown type, skip.")
            continue
        v_acc, v_pr, v_rc, v_f1, v_cm = evaluate_model(model, val_loader, device)
        t_acc, t_pr, t_rc, t_f1, t_cm = evaluate_model(model, test_loader, device)
        print(f"Val:  acc={v_acc:.4f} precision={v_pr:.4f} recall={v_rc:.4f} f1={v_f1:.4f}")
        print("Val confusion matrix:\n", v_cm)
        print(f"Test: acc={t_acc:.4f} precision={t_pr:.4f} recall={t_rc:.4f} f1={t_f1:.4f}")
        print("Test confusion matrix:\n", t_cm)
        results.append({
            "name": entry["name"],
            "val": (v_acc, v_pr, v_rc, v_f1, v_cm),
            "test": (t_acc, t_pr, t_rc, t_f1, t_cm)
        })
    print("\n=== Summary ===")
    header = "Model, ValAcc, ValPrec, ValRecall, ValF1, TestAcc, TestPrec, TestRecall, TestF1"
    print(header)
    for r in results:
        v = r["val"]; t = r["test"]
        print(f"{r['name']}, {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}, {v[3]:.4f}, {t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}, {t[3]:.4f}")

if __name__ == '__main__':
    main()