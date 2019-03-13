import numpy as np
import pandas as pd
import os.path as osp
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcNet(nn.Module):
    def __init__(self):
        super(ArcNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.classifier(x)
        x = torch.squeeze(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ArcNet().to(device)

train_label, train_value, test_value = utils.load_data()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 64

min_loss = 5

for epoch in range(10):
    index = np.arange(train_label.shape[0])
    np.random.shuffle(index)

    for i in range(0, index.shape[0] - batch_size, batch_size):
        input_data, input_label = train_value[index[i:i + batch_size]], train_label[index[i:i + batch_size]]
        input_data = np.reshape(input_data, [batch_size, 1, 28, 28])
        input_data, input_label = torch.from_numpy(input_data).float(), torch.from_numpy(input_label).long()
        input_data, input_label = input_data.to(device), input_label.to(device)
        optimizer.zero_grad()

        output = model(input_data)

        loss = criterion(output, input_label)
        loss.backward()

        loss_val = loss.item()
        if loss_val < min_loss:
            min_loss = loss_val
            torch.save(model.state_dict(), 'params.pt')
        acc = utils.calculate_acc(output.cpu().detach().numpy(), input_label.cpu().detach().numpy())
        print('loss: {:.3f}, acc: {:.3f}'.format(loss.item(), acc))

        optimizer.step()

model.load_state_dict(torch.load('params.pt'))
model.eval()
test_value = np.reshape(test_value, [-1, 1, 28, 28])
predictions = np.zeros(test_value.shape[0], dtype=np.int8)

print('test data size {}'.format(test_value.shape[0]))

for i in range(0, test_value.shape[0], 64):
    end_idx = min(test_value.shape[0], i + 64)
    batch = test_value[i:end_idx]
    batch = torch.from_numpy(batch).float().to(device)
    output = model(batch)
    output = output.cpu().detach().numpy()
    predictions[i:end_idx] = np.argmax(output, 1)
    print('{}'.format(end_idx), end='\r')
print('\nDone!')


utils.write_result(predictions.astype(np.int8))


# utils.visualize_data(test_value)







