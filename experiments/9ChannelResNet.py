import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torch.utils.data import random_split

import time
import random
from torch import nn

torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor()])

import torch
import torch.nn as nn
import os
import pickle


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            9, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x



def _resnet(arch, block, layers, pretrained, progress, device, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, **kwargs
    )


def evaluate_model(model, testloader_xy, testloader_xt, testloader_yt, device, criterion=None):
    """Evaluates the accuracy and loss for validation set
    Args:
    model: pruned model
    test_loader: Dataloader object containing Test data loading information
    device: Cuda or cpu device 
    criterion: method for calculating cross entropy loss
    Returns:
    eval_loss: Evaluated loss of the model
    eval_accuracy: Evaluated Accuracy of the model
    """
    model.eval()
    model.to(device=device)
    running_loss = 0.0
    running_corrects = 0.0

    inputs_xy = []
    inputs_xt = []
    inputs_yt = []
    labels_xy = []
    labels_xt = []
    labels_yt = []
    with torch.no_grad():
        for idx, item in enumerate(testloader_xy): 
            inputs_xy.append(item[0].to(device=device))
            labels_xy.append(item[1].to(device=device))

        for idx, item in enumerate(testloader_xt):
            inputs_xt.append(item[0].to(device=device))
            labels_xt.append(item[1].to(device=device))

        for idx, item in enumerate(testloader_yt): 
            inputs_yt.append(item[0].to(device=device))
            labels_yt.append(item[1].to(device=device))
        
        for i in range(len(testloader_xy)):
            optimizer.zero_grad()
            input = torch.cat((inputs_xy[i], inputs_xt[i], inputs_yt[i]), 1)
            outputs = model(input)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels_xy[i].long())
            
            # statistics
            running_loss += loss * inputs_xy[i].size(0)
            #print("preds == labels.data:", preds == labels.data)
            running_corrects += torch.sum(preds == labels_xy[i].data).float()
            #print(running_corrects)
        
        eval_loss = running_loss / len(testloader_xy.dataset)
        eval_accuracy = running_corrects / len(testloader_xy.dataset)
        return eval_loss, eval_accuracy

graphs = [{'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]} for k in range(12)]
dataset_xy = datasets.ImageFolder('gesture_data/data/IITM_DVS_10/features_extracted_xy', transform=transform)
dataset_xt = datasets.ImageFolder('gesture_data/data/IITM_DVS_10/features_extracted_xt', transform=transform)
dataset_yt = datasets.ImageFolder('gesture_data/data/IITM_DVS_10/features_extracted_yt', transform=transform)

for i in range(12):
    torch.cuda.empty_cache()
    model = resnet18(pretrained=False)
    if torch.cuda.is_available():
        model.cuda()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epochs = 50
    criterion = nn.CrossEntropyLoss()

    indices_list = list(range(1200))
    indices_to_remove = [] 
    for j in range(10):
            indices_to_remove = indices_to_remove + indices_list[120*j + 10*i:120*j + 10*i +10]

    indices_list = [e for e in indices_list if e not in indices_to_remove]
    random.shuffle(indices_list)
    random.shuffle(indices_to_remove)

    train_ds_xy = Subset(dataset_xy, indices_list)
    test_ds_xy = Subset(dataset_xy, indices_to_remove)

    print(len(train_ds_xy), len(test_ds_xy))
    
    train_ds_xt = Subset(dataset_xt, indices_list)
    test_ds_xt = Subset(dataset_xt, indices_to_remove)

    print(len(train_ds_xt), len(test_ds_xt))
    
    train_ds_yt = Subset(dataset_yt, indices_list)
    test_ds_yt = Subset(dataset_yt, indices_to_remove)

    trainloader_xy = torch.utils.data.DataLoader(train_ds_xy, batch_size=32, shuffle=False, num_workers=2)
    testloader_xy = torch.utils.data.DataLoader(test_ds_xy, batch_size=32, shuffle=False, num_workers=2)

    trainloader_xt = torch.utils.data.DataLoader(train_ds_xt, batch_size=32, shuffle=False, num_workers=2)
    testloader_xt = torch.utils.data.DataLoader(test_ds_xt, batch_size=32, shuffle=False, num_workers=2)

    trainloader_yt = torch.utils.data.DataLoader(train_ds_yt, batch_size=32, shuffle=False, num_workers=2)
    testloader_yt = torch.utils.data.DataLoader(test_ds_yt, batch_size=32, shuffle=False, num_workers=2)

    for epoch in range(num_epochs):
        print("epoch:",epoch)
        st = time.time()
        optimizer.zero_grad()

        running_loss = 0.0
        running_acc = 0.0
        inputs_xy = []
        inputs_xt = []
        inputs_yt = []
        labels_xy = []
        labels_xt = []
        labels_yt = []
        for idx, item in enumerate(trainloader_xy): 
            inputs_xy.append(item[0].to(device=device))
            labels_xy.append(item[1].to(device=device))

        for idx, item in enumerate(trainloader_xt): 
            inputs_xt.append(item[0].to(device=device))
            labels_xt.append(item[1].to(device=device))

        for idx, item in enumerate(trainloader_yt): 
            inputs_yt.append(item[0].to(device=device))
            labels_yt.append(item[1].to(device=device))
        for l in range(len(trainloader_xt)):
            input = torch.cat((inputs_xy[l], inputs_xt[l], inputs_yt[l]), 1)
            outputs = model(input)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels_xy[l].long())
            print(loss)
            #print("loss", loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            #print("running_loss", running_loss)
            running_acc += (torch.sum(preds == labels_xy[l].data).float() / preds.shape[0]).item()
            #print("running_accuracy", running_acc)

        graphs[i]['train_loss'].append(running_loss)
        graphs[i]['train_acc'].append(running_acc)
        test_loss, test_acc = evaluate_model(model, testloader_xy, testloader_xt, testloader_yt, device, criterion)
        graphs[i]['test_loss'].append(test_loss)
        graphs[i]['test_acc'].append(test_acc)
        print(test_loss, test_acc)

        # saving the trained model to a file
        torch.save(model.state_dict(), './gesture_modified_resnet3.pt')
        log = "{} {}/{} loss:{:.4f} acc:{:.4f}\n".format("phase", idx, len(trainloader_xy), running_loss / (idx+1), running_acc / (idx+1))
        print(log)
        print("time elapsed:", time.time()-st)
tot_acc = 0
for i in range(12):
    tot_acc += graphs[i]['test_acc']

print("mean accuracy: ", tot_acc/12)
f = open("./gesture_graph_modified_resnet3", "wb")
pickle.dump(graphs, f)


# import time
# from torch import nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# num_epochs = 10
# reg = 1e-4
# criterion = nn.CrossEntropyLoss()

# model_1.train()
# model_1.to(device=device)

# model_2.train()
# model_2.to(device=device)

# model_3.train()
# model_3.to(device=device)

# for epoch in range(num_epochs):
#     print("epoch:",epoch)
#     st = time.time()
#     running_loss = 0.0
#     running_acc = 0.0
#     for idx, item in enumerate(trainloader): 
#         #print(item[0][0].shape, item[1][0])       
#         inputs = item[0].to(device=device)
#         #print(item[0][0].shape, item[1][0])
#         labels = item[1].to(device=device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)
#         #print("loss", loss)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() 
#         #print("running_loss", running_loss)
#         running_acc += (torch.sum(preds == labels.data).float() / preds.shape[0]).item()
#         #print("running_accuracy", running_acc)
    
#     graphs['train_loss'].append(running_loss)
#     graphs['train_acc'].append(running_acc)
#     test_loss, test_acc = evaluate_model(model, testloader, device, criterion)
#     graphs['test_loss'].append(test_loss)
#     graphs['test_acc'].append(test_acc)
#     print(test_loss, test_acc)

#     # saving the trained model to a file
#     torch.save(model.state_dict(), '/content/drive/MyDrive/682\ Project/gesture_combined.pt')
#     log = "{} {}/{} loss:{:.4f} acc:{:.4f}\n".format("phase", idx, len(TrainLoader), running_loss / (idx+1), running_acc / (idx+1))
#     print(log)
#     print("time elapsed:", time.time()-st)