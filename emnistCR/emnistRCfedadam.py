import torchvision.transforms as transforms
import torch
import torchvision
import math
import torch.nn as nn
import numpy as np
import random
from cr import CR

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


def create_model_optimizer_criterion_dict(selected_client, learning_rate, momentum):
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    for i in selected_client:
        #构建模型
        model_name = "model" + str(i)
        model_info = CR()
        model_dict.update({model_name: model_info})
        #构建优化方法 此处使用SGD
        optimizer_name = "optimizer" + str(i)
        #optimizer_info = torch.optim.Adam(model_info.parameters(), lr=learning_rate)
        optimizer_info = torch.optim.SGD(model_info.parameters(), lr=learning_rate, momentum=momentum)
        optimizer_dict.update({optimizer_name: optimizer_info})
        #构建loss损失函数 此处使用crossEntropyloss
        criterion_name = "criterion" + str(i)
        criterion_info = nn.CrossEntropyLoss()
        criterion_dict.update({criterion_name: criterion_info})

    return model_dict, optimizer_dict, criterion_dict


def select_client(client_number, number_of_samples):
    selected_client = random.sample(range(number_of_samples), client_number)
    return selected_client

def send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, selected_client):
    with torch.no_grad():
        for i in selected_client:
            model_dict['model' + str(i)].conv1.weight.data = main_model.conv1.weight.data.clone()
            model_dict['model' + str(i)].conv1.bias.data = main_model.conv1.bias.clone()

            model_dict['model' + str(i)].conv2.weight.data = main_model.conv2.weight.data.clone()
            model_dict['model' + str(i)].conv2.bias.data = main_model.conv2.bias.data.clone()

            model_dict['model' + str(i)].dense.weight.data = main_model.dense.weight.data.clone()
            model_dict['model' + str(i)].dense.bias.data = main_model.dense.bias.data.clone()

            model_dict['model' + str(i)].dense2.weight.data = main_model.dense2.weight.data.clone()
            model_dict['model' + str(i)].dense2.bias.data = main_model.dense2.bias.data.clone()
    return model_dict

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
        v_t[name] = beta_2*v_t_1[name] + (1-beta_2)*real_delta_t_dict[name]**2
        with torch.no_grad():
            parameters.copy_(parameters+ torch.from_numpy(learning_rate*(real_delta_t_dict[name]/(np.sqrt(v_t[name])+tao))).to(device))
    return main_model, v_t, real_delta_t_dict

def start_train_end_node_process_without_print(x_train_dict, y_train_dict, beta_1, tao, learning_rate, learning_rate_of_clients, momentum, beta_2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    testset = torchvision.datasets.EMNIST(root='./emnist', train=False,
                                           download=True, transform=transform, split='byclass')
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)
    main_model = CR()
    main_model.to(device)
    #还没填充
    real_delta_t_dict = dict()
    v_t = dict()
    for name,parameters in main_model.named_parameters():
        real_delta_t_dict[name] = torch.zeros(size=parameters.cpu().data.shape).numpy()
        v_t[name] = torch.zeros(size=parameters.cpu().data.shape).numpy()

    for fd_epoch in range(10000):
        selected_client = select_client(10, 3400)
        model_dict, optimizer_dict, criterion_dict = create_model_optimizer_criterion_dict(selected_client, learning_rate_of_clients, momentum)
        model_dict = send_main_model_to_nodes_and_update_model_dict(main_model, model_dict, selected_client)
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
                print('fedepoch', fd_epoch, 'member', member, 'runningloss', running_loss, 'correct', correct, 'total', total)

        main_model, v_t, real_delta_t_dict = get_new_main_model(main_model, real_delta_t_dict, selected_client, model_dict , beta_1, beta_2, v_t, tao, learning_rate)
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

trainset = torchvision.datasets.EMNIST(root='./emnist', train=True, download=True, transform=torchvision.transforms.ToTensor(), split='byclass')
beta_1 = 0.9
beta_2 = 0.99
tao = 0.0001
momentum = 0.9
learning_rate = 0.03162
learning_rate_fd = 0.003162
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])
])

number_of_clients = 3400
train_dict = select_client(10, 3400)
image_data_dict, label_dict = get_label_and_images(number_of_clients, trainset)

#train_ds = torch.utils.data.TensorDataset(image_data_dict['images'+str(1)], label_dict['labels' + str(1)])
#train_dl = torch.utils.data.DataLoader(train_ds, batch_size=20, shuffle=True)
'''
for i, data in enumerate(train_dl, 0):
    images, labels = data
    print('image', images[0], images.shape)
    print('label', labels, labels.shape)

for i in range(len(image_data_dict)):
    print(i, label_dict['labels' + str(i)])
'''
start_train_end_node_process_without_print(image_data_dict, label_dict, beta_1, tao, learning_rate_fd, learning_rate, momentum, beta_2)