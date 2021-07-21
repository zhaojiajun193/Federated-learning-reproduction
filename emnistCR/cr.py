import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

class cr(nn.Module):
    def __init__(self):
        super(cr, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1, 1))
        self.relu = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dense2 = nn.Linear(128, 62)
        #self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpooling(out)
        out = self.dropout(out)
        out = self.flatten(out)
        out = self.dense(out)
        out = self.dropout2(out)
        #out = self.softmax(out)
        return F.log_softmax(self.dense2(out), dim=1)

def CR():
    return cr()

#下面的是测试
'''
model = CR()

learning_rate = 0.001
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
trainset = torchvision.datasets.EMNIST(root='./emnist', train=True, download=True, transform=torchvision.transforms.ToTensor(), split='mnist')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])
batch_size = 100
testset = torchvision.datasets.EMNIST(root='./emnist', train=False,download=True, transform=transform, split='mnist')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=0)
batch_size = 20
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
model = model.to(device)

for epoch in range(100):
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 不懂
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            #images = images.squeeze()
            #print(images.shape)
            #images = images.unsqueeze(3)
            #print(images.shape)
            # calculate outputs by running images through the network
            outputs = model(images)
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels.data).cpu().sum()
    print(correct, total, epoch, 'Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
            
'''



