from autoencoder import AE
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AE()
model.to(device)
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainset = torchvision.datasets.EMNIST(root='./emnist', train=True, download=True, transform=torchvision.transforms.ToTensor(), split='mnist')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])
batch_size_test = 100
testset = torchvision.datasets.EMNIST(root='./emnist', train=False,download=True, transform=transform, split='mnist')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,shuffle=False, num_workers=0)
batch_size = 20
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
print(len(trainset))

def get_label_and_images(number_of_clients, trainset):
    image_data_dict = dict()
    label_dict = dict()
    each_client_have = int(math.floor(len(trainset)/number_of_clients))
    for i in range(number_of_clients):
        image_name = 'images' + str(i)
        label_name = 'labels' + str(i)
        labels = []
        q = 0
        start_index = i*each_client_have
        for j in range(start_index, start_index + each_client_have):
            image, label = trainset[j]
            labels.append(label)
            print(j)
            if q == 0:
                images = image.clone().detach()
            elif q == 1:
                images = torch.stack((images, image), dim=0)
            else:
                image = image.unsqueeze(0)
                images = torch.cat((images, image), dim=0)
            q = q + 1
        image_data_dict.update({image_name:images})
        labels = torch.tensor(labels)
        label_dict.update({label_name:labels})
    return image_data_dict, label_dict
print('开始')
image_data_dict, label_dict = get_label_and_images(500, trainset)
train_ds = torch.utils.data.TensorDataset(image_data_dict['images'+str(0)], label_dict['labels' + str(0)])
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=20, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=20,shuffle=True, num_workers=0)
'''
image_1, label_1 = trainset[0]
image_2, label_2 = train_ds[0]
for data in train_dl:
    images, labels = data
    print(images.shape)

for data in trainloader:
    images, labels = data
    print(images.shape)
'''
for epoch in range(10000):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        inputs = inputs.reshape(len(inputs), 28*28).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 不懂
    print(epoch, running_loss)
    print(epoch, '更新', model.encoder[0].weight.data)
    main_model_loss = nn.MSELoss()
    '''
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.reshape(len(images), 28 * 28).to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            loss = main_model_loss(outputs, images)
            running_loss += loss.item()
    print(epoch, running_loss, running_loss / len(testset))
    '''
    if epoch%100==0:
        toPIL = transforms.ToPILImage()
        img, label = train_ds[0]
        img = toPIL(img)
        img.save('r123.jpg')
        img, label = train_ds[0]
        img = img.to(device)
        out = model(img.reshape(28*28)).reshape(1,28,28)
        out = toPIL(out)
        out.save(str(epoch)+'.jpg')
