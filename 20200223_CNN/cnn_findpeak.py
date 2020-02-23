# -*- coding: utf-8 -*-

import glob
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

# フォルダ testfig においた画像ファイルのリストを取得
image_path = "./testfig"
files = glob.glob(image_path + "/*.png")

x = []
y = []

for filename in files:
    # 画像ファイルをndarrayに変換する
    im = Image.open(filename)
    im = im.convert("1")  # モノクロ画像に変換
    im = im.resize((28, 28))  # 28x28にリサイズ
    im = np.array(im).astype(int)  # intのarrayに変換
    x.append([im])  # CNNの入力に合うshapeとなるよう注意

    # ファイル名から正解ラベルを取得
    label = int(filename[len(image_path)+1])
    y.append(label)

x = np.array(x)  # x.shape (画像数, 1, 28, 28)
y = np.array(y)  # y.shape (画像数,)


# 訓練データとテストデータに分ける
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.2, shuffle=False)

# テンソルに変換
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

# Dataloaderを準備
train_dataloader = torch.utils.data.TensorDataset(x_train, y_train)
train_dataloader = torch.utils.data.DataLoader(train_dataloader, batch_size=4)


# ニューラルネットワーク定義
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # チャンネル1, フィルタ6, フィルタサイズ5x5
        self.pool1 = nn.MaxPool2d(2, 2)  # 2x2のプーリング層
        self.conv2 = nn.Conv2d(6, 16, 5)  # チャンネル6, フィルタ16, フィルタサイズ5x5
        self.pool2 = nn.MaxPool2d(2, 2)

        # これまでの畳込みとプーリングで、16チャンネルの4x4が入力サイズ
        self.fc1 = nn.Linear(16 * 4 * 4, 64)  # fc: fully connected
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 16 * 4 * 4)  # [(バッチサイズ), (一次元配列データ)]に並び替え
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 学習
model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

num_epochs = 20

for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print('Epoch: {}  Loss: {}'.format(epoch, total_loss))


#  検証(正答率の確認)
outputs = model(x_test)
_, predicted = torch.max(outputs.data, 1)
correct = (y_test == predicted).sum().item()  # Tensorの比較で正答数を確認
print("正答率: {} %".format(correct/len(predicted)*100.0))
