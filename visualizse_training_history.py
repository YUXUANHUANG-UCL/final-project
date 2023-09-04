import matplotlib.pyplot as plt

# 读取训练历史数据
log_file_path = "/home/uceeuam/graduation_project/checkpoints_pretrain/log.txt"
with open(log_file_path, 'r') as f:
    lines = f.readlines()

# 初始化变量
train_epochs = []
val_epochs = []
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 解析训练历史数据
epoch_data = {}
for line in lines:
    if line.startswith('Epoch'):
        parts = line.split(', ')
        epoch = int(parts[0].split(': ')[1])
        iteration = int(parts[1].split(': ')[1])
        metric_type = parts[2].split(': ')[0]
        metric_value = float(parts[2].split(': ')[1])

        if epoch not in epoch_data:
            epoch_data[epoch] = {}

        if metric_type == 'Train/loss':
            epoch_data[epoch]['train_loss'] = metric_value
        elif metric_type == 'Train/accuracy':
            epoch_data[epoch]['train_accuracy'] = metric_value
        elif metric_type == 'Eval/loss':
            epoch_data[epoch]['val_loss'] = metric_value
        elif metric_type == 'Eval/accuracy':
            epoch_data[epoch]['val_accuracy'] = metric_value
# 提取数据用于绘图
for epoch, data in epoch_data.items():
    if 'train_loss' in data.keys():
        train_epochs.append(epoch)
        train_losses.append(data['train_loss'])
        train_accuracies.append(data['train_accuracy'])
    if 'val_loss' in data.keys():
        val_epochs.append(epoch)
        val_losses.append(data['val_loss'])
        val_accuracies.append(data['val_accuracy'])

# 寻找最低 Loss 和最高 Accuracy 的位置
min_loss_epoch_val = val_epochs[val_losses.index(min(val_losses[:-1]))]
max_acc_epoch_val = val_epochs[val_accuracies.index(max(val_accuracies[:-1]))]
min_loss_epoch_train = train_epochs[train_losses.index(min(train_losses))]
max_acc_epoch_train = train_epochs[train_accuracies.index(max(train_accuracies))]

# 绘制图形
plt.figure(figsize=(10, 6))

# 绘制训练和验证Loss曲线
plt.subplot(2, 1, 1)
plt.plot(train_epochs, train_losses, label='Train Loss')
plt.plot(val_epochs[:-2], val_losses[:-2], label='Validation Loss')
plt.scatter(min_loss_epoch_train, min(train_losses), color='blue', label='Min Train Loss')
plt.scatter(min_loss_epoch_val, min(val_losses), color='red', label='Min Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0)  # 将x轴范围限制在大于等于0的部分
plt.legend()

# 绘制垂直线并标注最低 Loss 的值
plt.annotate(f'{min(val_losses):.4f}', xy=(0, min(val_losses)),
             xytext=(-60, -7), textcoords='offset points',
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.plot([0, min_loss_epoch_val], [min(val_losses), min(val_losses)], color='black', linestyle='dashed')

# 绘制垂直线并标注最低 Train Loss 的值
plt.annotate(f'{min(train_losses):.4f}', xy=(0, min(train_losses)),
             xytext=(-60, -3), textcoords='offset points',
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.plot([0, min_loss_epoch_train], [min(train_losses), min(train_losses)], color='black', linestyle='dashed')

# 绘制训练和验证Accuracy曲线
plt.subplot(2, 1, 2)
plt.plot(train_epochs, train_accuracies, label='Train Accuracy')
plt.plot(val_epochs[:-2], val_accuracies[:-2], label='Validation Accuracy')
plt.scatter(max_acc_epoch_train, max(train_accuracies), color='green', label='Max Train Accuracy')
plt.scatter(max_acc_epoch_val, max(val_accuracies), color='orange', label='Max Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlim(0)  # 将x轴范围限制在大于等于0的部分
plt.legend()

# 绘制垂直线并标注最高 Accuracy 的值
plt.annotate(f'{max(val_accuracies):.2f}', xy=(0, max(val_accuracies)),
             xytext=(-60, -3), textcoords='offset points',
             arrowprops=dict(facecolor='black', arrowstyle='->'))
print(max_acc_epoch_val)
plt.plot([0, max_acc_epoch_val], [max(val_accuracies), max(val_accuracies)], color='black', linestyle='dashed')

# 绘制垂直线并标注最高 Train Accuracy 的值
plt.annotate(f'{max(train_accuracies):.2f}', xy=(0, max(train_accuracies)),
             xytext=(-60, -3), textcoords='offset points',
             arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.plot([0, max_acc_epoch_train], [max(train_accuracies), max(train_accuracies)], color='black', linestyle='dashed')

plt.tight_layout()
plt.savefig('fig/trainval_pretrain.jpg')

