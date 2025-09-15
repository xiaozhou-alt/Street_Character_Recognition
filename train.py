import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 参数配置
class Config:
    batch_size = 32  # 减小批处理大小以适应更大模型
    num_epochs = 100  # 增加训练轮数
    learning_rate = 0.0005  # 使用稍高的学习率
    patience = 10  # 增加耐心值
    image_size = (64, 256)  # 增加宽度以适应更多字符
    max_seq_length = 10  # 增加最大序列长度
    num_classes = 10  # 0-9数字
    weight_decay = 1e-5  # 减小权重衰减
    train_data_ratio = 1.0  # 使用全部训练数据
    dropout = 0.3  # 添加dropout
    warmup_steps = 1000  # 学习率预热步骤
    transformer_layers = 3  # Transformer层数
    num_heads = 8  # Transformer头数

config = Config()

# 自定义变换：保持宽高比的填充和调整大小
class AspectRatioPreservingResize:
    def __init__(self, target_size):
        self.target_size = target_size  # (height, width)
    
    def __call__(self, img):
        # 获取原始尺寸
        original_width, original_height = img.size
        
        # 计算缩放比例
        scale = min(self.target_size[0] / original_height, 
                   self.target_size[1] / original_width)
        
        # 计算新尺寸
        new_height = int(original_height * scale)
        new_width = int(original_width * scale)
        
        # 调整大小
        img = img.resize((new_width, new_height), Image.BILINEAR)
        
        # 创建新图像并填充
        new_img = Image.new('RGB', (self.target_size[1], self.target_size[0]), (0, 0, 0))
        
        # 计算粘贴位置（居中）
        paste_x = (self.target_size[1] - new_width) // 2
        paste_y = (self.target_size[0] - new_height) // 2
        
        # 粘贴图像
        new_img.paste(img, (paste_x, paste_y))
        
        return new_img

# 数据增强变换
train_transform = transforms.Compose([
    AspectRatioPreservingResize(config.image_size),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    AspectRatioPreservingResize(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据预处理和加载
class StreetViewDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, is_train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        
        # 加载标注文件
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # 获取所有图像文件名
        self.img_names = list(self.annotations.keys())
        
        # 如果是训练模式，过滤掉没有标注的图像
        if is_train:
            self.img_names = [img_name for img_name in self.img_names 
                             if 'label' in self.annotations[img_name] and 
                             len(self.annotations[img_name]['label']) > 0]
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 如果图像损坏，创建一个空白图像
            image = Image.new('RGB', config.image_size[::-1], (0, 0, 0))
        
        # 获取标注信息
        ann = self.annotations[img_name]
        
        if self.is_train:
            # 对于训练数据，提取字符区域并创建标签
            labels = ann['label']
            lefts = ann['left']
            tops = ann['top']
            widths = ann['width']
            heights = ann['height']
            
            # 按从左到右的顺序排序字符
            sorted_indices = np.argsort(lefts)
            labels = [labels[i] for i in sorted_indices]
            
            # 创建序列标签（填充到最大长度）
            seq_label = torch.full((config.max_seq_length,), -1, dtype=torch.long)
            for i, label in enumerate(labels[:config.max_seq_length]):
                seq_label[i] = label
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, seq_label, len(labels)
        else:
            # 对于测试数据，只返回图像和文件名
            if self.transform:
                image = self.transform(image)
            
            return image, img_name

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 基于EfficientNet-B7和Transformer的模型
class EfficientNetB7Transformer(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=3, nhead=8):
        super(EfficientNetB7Transformer, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # 使用预训练的EfficientNet-B7作为CNN特征提取器
        self.cnn = models.efficientnet_b7(pretrained=True).features
        
        # 计算CNN输出特征图的大小
        with torch.no_grad():
            sample = torch.randn(1, 3, config.image_size[0], config.image_size[1])
            cnn_out = self.cnn(sample)
            self.cnn_output_size = cnn_out.size(1)
            self.seq_length = cnn_out.size(3)
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # 减少通道数的卷积层
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(self.cnn_output_size, hidden_size, kernel_size=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config.dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=config.dropout,
            activation='gelu'
        )
        
        # Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_size, num_classes + 1)  # +1 for CTC blank token
        
        # 初始化权重
        self.init_weights()
    
    def init_weights(self):
        # 初始化输出层
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
        # 初始化卷积层
        for m in self.conv_reduce.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN特征提取
        cnn_features = self.cnn(x)
        
        # 自适应池化
        cnn_features = self.adaptive_pool(cnn_features)
        
        # 减少通道数
        cnn_features = self.conv_reduce(cnn_features)
        
        # 重塑特征以适应Transformer输入
        batch_size, channels, height, width = cnn_features.size()
        cnn_features = cnn_features.squeeze(2)  # 移除高度维度
        cnn_features = cnn_features.permute(2, 0, 1)  # (seq_len, batch, hidden_size)
        
        # 添加位置编码
        cnn_features = self.pos_encoder(cnn_features)
        
        # Transformer处理
        transformer_out = self.transformer_encoder(cnn_features)
        
        # 输出层
        output = self.fc(transformer_out)
        output = output.permute(1, 0, 2)  # (batch, seq_len, num_classes+1)
        output = output.log_softmax(2)  # CTC需要log softmax
        
        return output

# CTC损失函数
def ctc_loss_fn(predictions, targets, target_lengths):
    # 确保target_lengths是正确形状的张量
    target_lengths = torch.tensor(target_lengths, dtype=torch.long).cpu()
    
    # 处理targets - 移除填充值(-1)
    targets_flat = []
    for i in range(targets.size(0)):
        valid_targets = targets[i][targets[i] != -1]
        targets_flat.append(valid_targets)
    
    # 将所有有效目标连接成一个一维张量
    targets_concat = torch.cat(targets_flat).cpu()
    
    # 创建输入长度张量
    input_lengths = torch.full(
        size=(predictions.size(0),), 
        fill_value=predictions.size(1), 
        dtype=torch.long
    ).cpu()
    
    # 调整预测张量的维度以适应CTC损失
    predictions = predictions.permute(1, 0, 2)  # (T, N, C)
    
    # 计算CTC损失
    loss = nn.CTCLoss(blank=config.num_classes, zero_infinity=True)(
        predictions, targets_concat, input_lengths, target_lengths
    )
    
    return loss

# 解码预测结果
def decode_predictions(predictions):
    """将模型输出转换为数字序列"""
    _, max_indices = torch.max(predictions, 2)
    decoded_seqs = []
    
    for seq in max_indices:
        # 移除重复和空白标记
        prev_char = -1
        decoded_seq = []
        for char_idx in seq:
            if char_idx != prev_char and char_idx != config.num_classes:  # 忽略空白标记
                decoded_seq.append(char_idx.item())
            prev_char = char_idx
        
        decoded_seqs.append(decoded_seq)
    
    return decoded_seqs

# 计算准确率
def calculate_accuracy(predictions, targets, target_lengths):
    decoded_seqs = decode_predictions(predictions)
    correct = 0
    total = len(targets)
    
    for i in range(total):
        pred_seq = decoded_seqs[i]
        # 获取真实标签（移除填充值-1）
        true_seq = targets[i][targets[i] != -1].cpu().numpy()
        
        if len(pred_seq) == len(true_seq) and np.array_equal(pred_seq, true_seq):
            correct += 1
    
    return correct / total if total > 0 else 0

# 学习率预热
def warmup_lr(optimizer, step, warmup_steps, base_lr):
    lr = base_lr * min(step / warmup_steps, 1.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# 训练函数
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, config):
    model.to(device)
    best_accuracy = 0.0
    patience_counter = 0
    
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    # 学习率预热步骤
    step = 0
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs} [Train]')
        for images, targets, target_lengths in pbar:
            images, targets = images.to(device), targets.to(device)
            
            # 学习率预热
            if step < config.warmup_steps:
                current_lr = warmup_lr(optimizer, step, config.warmup_steps, config.learning_rate)
                step += 1
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 检查输出是否有NaN
            if torch.isnan(outputs).any():
                print("输出包含NaN，跳过该批次")
                continue
            
            # 计算CTC损失
            loss = ctc_loss_fn(outputs, targets, target_lengths)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print("损失为NaN，跳过该批次")
                continue
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets, target_lengths in val_loader:
                images, targets = images.to(device), targets.to(device)
                
                outputs = model(images)
                loss = ctc_loss_fn(outputs, targets, target_lengths)
                
                # 检查损失是否为NaN
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    
                    accuracy = calculate_accuracy(outputs, targets, target_lengths)
                    val_accuracy += accuracy
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_accuracy = val_accuracy / val_batches if val_batches > 0 else 0
        
        # 更新学习率
        scheduler.step(avg_val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_accuracy)
        history['learning_rate'].append(current_lr)
        
        print(f'Epoch {epoch+1}/{config.num_epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Accuracy: {avg_val_accuracy:.4f}, '
              f'LR: {current_lr:.6f}')
        
        # 保存最佳模型
        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv('/kaggle/working/training_history.csv', index=False)
    print("Training history saved to 'training_history.csv'")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Training Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/training_curves.png')
    plt.show()
    
    return best_accuracy, history_df

# 测试随机样本
def test_random_samples(model, val_loader, device, num_samples=10):
    model.eval()
    samples = []
    
    # 收集样本
    with torch.no_grad():
        for images, targets, target_lengths in val_loader:
            images, targets = images.to(device), targets.to(device)
            
            outputs = model(images)
            decoded_seqs = decode_predictions(outputs)
            
            for i in range(len(images)):
                if len(samples) >= num_samples:
                    break
                
                pred_seq = decoded_seqs[i]
                # 获取真实标签（移除填充值-1）
                true_seq = targets[i][targets[i] != -1].cpu().numpy()
                
                # 将数字序列转换为字符串
                pred_str = ''.join(str(x) for x in pred_seq)
                true_str = ''.join(str(x) for x in true_seq)
                
                samples.append({
                    'image': images[i].cpu(),
                    'prediction': pred_str,
                    'true_label': true_str,
                    'correct': pred_str == true_str
                })
            
            if len(samples) >= num_samples:
                break
    
    # 显示样本
    print("\n随机样本测试结果:")
    print("=" * 50)
    
    for i, sample in enumerate(samples):
        print(f"样本 {i+1}:")
        print(f"  预测: {sample['prediction']}")
        print(f"  真实: {sample['true_label']}")
        print(f"  正确: {sample['correct']}")
        print("-" * 30)
    
    # 计算准确率
    correct_count = sum(1 for sample in samples if sample['correct'])
    accuracy = correct_count / len(samples)
    print(f"\n随机样本准确率: {accuracy:.2%}")

# 预测测试集
def predict_test_set(model, test_loader, device, output_file='/kaggle/working/submission.csv'):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="预测测试集"):
            images = images.to(device)
            outputs = model(images)
            decoded_seqs = decode_predictions(outputs)
            
            for i in range(len(images)):
                pred_seq = decoded_seqs[i]
                pred_str = ''.join(str(x) for x in pred_seq)
                
                # 处理文件名扩展名
                img_name = img_names[i]
                if img_name.endswith('.png'):
                    img_name = img_name.replace('.png', '.jpg')
                
                results.append({
                    'file_name': img_name,
                    'file_code': pred_str
                })
    
    # 创建提交文件
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_file, index=False)
    print(f"提交文件已保存到: {output_file}")
    
    return submission_df

# 主函数
def main():
    # 数据路径
    train_img_dir = '/kaggle/input/street-character-recognition/data/mchar_train'  # 修改为您的训练集路径
    val_img_dir = '/kaggle/input/street-character-recognition/data/mchar_val'      # 修改为您的验证集路径
    test_img_dir = '/kaggle/input/street-character-recognition/data/mchar_test_a'    # 修改为您的测试集路径
    
    train_ann_file = '/kaggle/input/street-character-recognition/data/mchar_train.json'  # 修改为您的训练标注文件路径
    val_ann_file = '/kaggle/input/street-character-recognition/data/mchar_val.json'      # 修改为您的验证标注文件路径
    
    # 创建数据集
    print("加载训练数据...")
    train_dataset = StreetViewDataset(train_img_dir, train_ann_file, train_transform, is_train=True)
    
    print("加载验证数据...")
    val_dataset = StreetViewDataset(val_img_dir, val_ann_file, val_transform, is_train=True)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4,  # 增加工作线程数
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,  # 增加工作线程数
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 初始化模型
    print("初始化模型...")
    model = EfficientNetB7Transformer(
        config.num_classes, 
        hidden_size=512,
        num_layers=config.transformer_layers,
        nhead=config.num_heads
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 定义优化器和学习率调度器
    optimizer = optim.AdamW(  # 使用AdamW优化器
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True  # 根据准确率调整学习率
    )
    
    # 训练模型
    print("开始训练模型...")
    best_accuracy, history_df = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device, config
    )
    print(f"最佳验证准确率: {best_accuracy:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('/kaggle/working/best_model.pth'))
    
    # 测试随机样本
    test_random_samples(model, val_loader, device, num_samples=10)
    
    # 预测测试集
    print("加载测试数据...")
    test_dataset = StreetViewDataset(test_img_dir, val_ann_file, val_transform, is_train=False)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    submission_df = predict_test_set(model, test_loader, device, '/kaggle/working/submission.csv')
    print(f"生成 {len(submission_df)} 条预测结果")

if __name__ == "__main__":
    main()