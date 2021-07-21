import torchvision.transforms as transforms
import torch
from resnet import ResNet18
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
import math
import random

def split_and_shuffle_labels(y_data, seed):
    y_data = pd.DataFrame(y_data, columns=['labels'])
    y_data['i'] = np.arange(len(y_data))
    label_dict = dict()
    for i in range(10):
        var_name = 'label' + str(i)
        label_info = y_data[y_data['labels']==i]
        np.random.seed(seed)
        label_info = np.random.permutation(label_info)
        label_info = pd.DataFrame(label_info, columns=['labels','i'])
        label_dict.update({var_name:label_info})
    return label_dict


def get_iid_subsamples_indices(label_dict, number_of_samples):
    #构建一个字典
    sample_dict= dict()
    #math.floor向下取整 用于计算每个手机分配多少个样本
    batch_size=int(math.floor(5000/number_of_samples))
    for i in range(number_of_samples):
        sample_name="sample"+str(i)
        #构建一个excel表
        dumb=pd.DataFrame()
        for j in range(10):
            label_name=str("label")+str(j)
            a=label_dict[label_name][i*batch_size:(i+1)*batch_size]
            dumb=pd.concat([dumb,a], axis=0)
        dumb.reset_index(drop=True, inplace=True)
        sample_dict.update({sample_name: dumb})
    return sample_dict

def create_iid_subsamples(sample_dict, trainset):
    x_data_dict = dict()
    #图片数据
    y_data_dict = dict()
    #标签数据
    for i in range(len(sample_dict)):  ### len(sample_dict)= number of samples

        xname = 'images' + str(i)
        yname = 'labels' + str(i)
        sample_name = "sample" + str(i)
        #找到sample_dict对应的手机分配里边的index sort从小到大排序
        q = 0
        labels = []
        for j in sample_dict[sample_name]["i"].values.tolist():
            image, label = trainset[j]
            labels.append(label)
            if q == 0:
                images = image.clone().detach()
            elif q == 1:
                images = torch.stack((images, image), dim=0)
            else:
                image = image.unsqueeze(0)
                images = torch.cat((images, image), dim=0)
            q = q + 1
        x_data_dict.update({xname: images})
        labels = torch.tensor(labels)
        y_data_dict.update({yname: labels})

    return x_data_dict, y_data_dict

def select_client(client_number, number_of_samples):
    selected_client = random.sample(range(number_of_samples), client_number)
    return selected_client

def get_new_main_model(main_model, last_delta_t, samples, model_dict_2, beta_1, beta_2, v_t_1, tao, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    delta_t_dict = dict()
    real_delta_t_dict = dict()
    v_t = dict()
    main_model_dict = dict()
    for name,parameters in main_model.named_parameters():
        real_delta_t_dict[name] = torch.zeros(size=parameters.cpu().data.shape).numpy()
        main_model_dict.update({name:parameters.cpu().detach().numpy()})
    for i in samples:
        model_name = 'model' + str(i)
        model_1 = main_model
        model_2 = model_dict_2[model_name]
        parm_old = dict()
        parm_new = dict()
        parm_zero = dict()
        for name, parameters in model_1.named_parameters():
            parm_old.update({name:parameters.cpu().detach().numpy()})
            parm_zero.update({name:torch.zeros(size=parameters.cpu().data.shape).numpy()})
        for name, parameters in model_2.named_parameters():
            parm_new.update({name:parameters.cpu().detach().numpy()})
            parm_zero[name] = parm_new[name] - parm_old[name]
        delta_t_dict.update({model_name:parm_zero})
    #last_delta_t和delta_t_dict格式相同

    for name,parameters in main_model.named_parameters():
        real_delta_t_dict[name] = torch.zeros(size=parameters.cpu().data.shape).numpy()
        main_model_dict.update({name:parameters.cpu().detach().numpy()})
        for i in samples:
            real_delta_t_dict[name] += delta_t_dict['model'+str(i)][name]
        real_delta_t_dict[name] = real_delta_t_dict[name]/len(samples)
        real_delta_t_dict[name] = beta_1*last_delta_t[name] + (1-beta_1)*real_delta_t_dict[name]
    #last_delta_t和real_delta_t_dict格式相同
        v_t[name] = v_t_1[name] - ((1-beta_2)*real_delta_t_dict[name]**2)*(np.sign(v_t_1[name]-real_delta_t_dict[name]**2))
        with torch.no_grad():
            parameters.copy_(parameters+ torch.from_numpy(learning_rate*(real_delta_t_dict[name]/(np.sqrt(v_t[name])+tao))).to(device))
    return main_model, v_t, real_delta_t_dict

def create_model_optimizer_criterion_dict(selected_client, learning_rate, momentum):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    for i in selected_client:
        #构建模型
        model_name = "model" + str(i)
        model_info = ResNet18()
        model_dict.update({model_name: model_info})
        #构建优化方法 此处使用SGD
        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})
        #构建loss损失函数 此处使用crossEntropyloss
        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict

def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, selected_client):
    with torch.no_grad():
        for i in selected_client:
            model_dict['model' + str(i)].conv1.weight.data = main_model.conv1.weight.data.clone()

            model_dict['model' + str(i)].bn1.weight.data = main_model.bn1.weight.data.clone()
            model_dict['model' + str(i)].bn1.bias.data = main_model.bn1.bias.data.clone()

            model_dict['model' + str(i)].layer1[0].conv1.weight.data = main_model.layer1[0].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer1[0].bn1.weight.data = main_model.layer1[0].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer1[0].bn1.bias.data = main_model.layer1[0].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer1[0].conv2.weight.data = main_model.layer1[0].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer1[0].bn2.weight.data = main_model.layer1[0].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer1[0].bn2.bias.data = main_model.layer1[0].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer1[1].conv1.weight.data = main_model.layer1[1].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer1[1].bn1.weight.data = main_model.layer1[1].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer1[1].bn1.bias.data = main_model.layer1[1].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer1[1].conv2.weight.data = main_model.layer1[1].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer1[1].bn2.weight.data = main_model.layer1[1].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer1[1].bn2.bias.weight = main_model.layer1[1].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer2[0].conv1.weight.data = main_model.layer2[0].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer2[0].bn1.weight.data = main_model.layer2[0].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer2[0].bn1.bias.data = main_model.layer2[0].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer2[0].conv2.weight.data = main_model.layer2[0].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer2[0].bn2.weight.data = main_model.layer2[0].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer2[0].bn2.bias.data = main_model.layer2[0].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer2[0].shortcut[0].weight.data = main_model.layer2[0].shortcut[
                0].weight.data.clone()

            model_dict['model' + str(i)].layer2[0].shortcut[1].weight.data = main_model.layer2[0].shortcut[
                1].weight.data.clone()
            model_dict['model' + str(i)].layer2[0].shortcut[1].bias.data = main_model.layer2[0].shortcut[
                1].bias.data.clone()

            model_dict['model' + str(i)].layer2[1].conv1.weight.data = main_model.layer2[1].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer2[1].bn1.weight.data = main_model.layer2[1].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer2[1].bn1.bias.data = main_model.layer2[1].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer2[1].conv2.weight.data = main_model.layer2[1].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer2[1].bn2.weight.data = main_model.layer2[1].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer2[1].bn2.bias.data = main_model.layer2[1].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer3[0].conv1.weight.data = main_model.layer3[0].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer3[0].bn1.weight.data = main_model.layer3[0].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer3[0].bn1.bias.data = main_model.layer3[0].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer3[0].conv2.weight.data = main_model.layer3[0].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer3[0].bn2.weight.data = main_model.layer3[0].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer3[0].bn2.bias.data = main_model.layer3[0].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer3[0].shortcut[0].weight.data = main_model.layer3[0].shortcut[
                0].weight.data.clone()

            model_dict['model' + str(i)].layer3[0].shortcut[1].weight.data = main_model.layer3[0].shortcut[
                1].weight.data.clone()
            model_dict['model' + str(i)].layer3[0].shortcut[1].bias.data = main_model.layer3[0].shortcut[
                1].bias.data.clone()

            model_dict['model' + str(i)].layer3[1].conv1.weight.data = main_model.layer3[1].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer3[1].bn1.weight.data = main_model.layer3[1].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer3[1].bn1.bias.data = main_model.layer3[1].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer3[1].conv2.weight.data = main_model.layer3[1].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer3[1].bn2.weight.data = main_model.layer3[1].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer3[1].bn2.bias.data = main_model.layer3[1].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer4[0].conv1.weight.data = main_model.layer4[0].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer4[0].bn1.weight.data = main_model.layer4[0].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer4[0].bn1.bias.data = main_model.layer4[0].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer4[0].conv2.weight.data = main_model.layer4[0].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer4[0].bn2.weight.data = main_model.layer4[0].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer4[0].bn2.bias.data = main_model.layer4[0].bn2.bias.data.clone()

            model_dict['model' + str(i)].layer4[0].shortcut[0].weight.data = main_model.layer4[0].shortcut[
                0].weight.data.clone()

            model_dict['model' + str(i)].layer4[0].shortcut[1].weight.data = main_model.layer4[0].shortcut[
                1].weight.data.clone()
            model_dict['model' + str(i)].layer4[0].shortcut[1].bias.data = main_model.layer4[0].shortcut[
                1].bias.data.clone()

            model_dict['model' + str(i)].layer4[1].conv1.weight.data = main_model.layer4[1].conv1.weight.data.clone()

            model_dict['model' + str(i)].layer4[1].bn1.weight.data = main_model.layer4[1].bn1.weight.data.clone()
            model_dict['model' + str(i)].layer4[1].bn1.bias.data = main_model.layer4[1].bn1.bias.data.clone()

            model_dict['model' + str(i)].layer4[1].conv2.weight.data = main_model.layer4[1].conv2.weight.data.clone()

            model_dict['model' + str(i)].layer4[1].bn2.weight.data = main_model.layer4[1].bn2.weight.data.clone()
            model_dict['model' + str(i)].layer4[1].bn2.bias.data = main_model.layer4[1].bn2.bias.data.clone()

            model_dict['model' + str(i)].linear.weight.data = main_model.linear.weight.data.clone()
            model_dict['model' + str(i)].linear.bias.data = main_model.linear.bias.data.clone()

    return model_dict

def start_train_end_node_process_without_print(x_train_dict, y_train_dict, beta_1, beta_2, tao, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    main_model = ResNet18()
    main_model.to(device)
    #还没填充
    real_delta_t_dict = dict()
    v_t = dict()
    for name,parameters in main_model.named_parameters():
        real_delta_t_dict[name] = torch.zeros(size=parameters.cpu().data.shape).numpy()
        v_t[name] = torch.zeros(size=parameters.cpu().data.shape).numpy()

    for fd_epoch in range(10000):
        selected_client = select_client(10, 500)
        model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(selected_client, learning_rate, momentum)
        model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, selected_client)
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for member in selected_client:

            train_ds = torch.utils.data.TensorDataset(x_train_dict['images'+str(member)], y_train_dict['labels'+str(member)])
            train_dl = torch.utils.data.DataLoader(train_ds, batch_size=20, shuffle=True)


            model = model_dict['model' + str(member)]
            #for name, parameters in model.named_parameters():
                #print(name, parameters.data.shape)
            model.to(device)
            criterion = criterion_dict["criterion" + str(member)]
            optimizer = optimizer_dict["optimizer" + str(member)]

            for epoch in range(1):
                running_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(train_dl, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    length = len(train_dl)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    # 不懂
                    total += labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.data).cpu().sum()
                    model_dict.update({'model' + str(member):model})
                    #print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                    #      % (epoch + 1, (i + 1 + epoch * length), running_loss / (i + 1), 100. * correct / total))
                correct = 0
                total = 0

        main_model, v_t, real_delta_t_dict = get_new_main_model(main_model, real_delta_t_dict, selected_client,model_dict , beta_1, beta_2, v_t, tao, learning_rate)
        #print('更新',main_model.conv1.weight.data)
        #print('vt',v_t['conv1.weight'])
        #print('deltat', real_delta_t_dict['conv1.weight'])
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = main_model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(correct, total, fd_epoch,'fdlearning Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

seed = 1
number_of_samples = 500
select_number = 10
learning_rate = 0.03162
learning_rate_fd = 0.01
momentum = 0.9
beta_1 = 0.9
beta_2 = 0.99
tao = 0.001
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 100
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
labels = []
for i in range(len(trainset)):
    images, label = trainset[i]
    labels.append(label)
label_dict = split_and_shuffle_labels(labels, seed)
sample_dict = get_iid_subsamples_indices(label_dict, number_of_samples)
x_data_dict, y_data_dict = create_iid_subsamples(sample_dict, trainset)
start_train_end_node_process_without_print(x_data_dict, y_data_dict, beta_1, beta_2, tao, learning_rate_fd)