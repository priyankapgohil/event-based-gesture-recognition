import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torchvision.models as models
import time
from torch import nn
import pickle


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor()])

dataset = datasets.ImageFolder('gesture_data/data/IITM_DVS_10/features_extracted_xy', transform=transform)

train_size = 867
test_size = 333

train_ds, test_ds = random_split(dataset, [train_size, test_size])
print(len(train_ds), len(test_ds))

# train_ds, val_ds = random_split(train_ds, [2000, 600])
# print(len(train_ds), len(val_ds))

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=2)
#valloader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=True, num_workers=2)

model = models.resnet18()

def evaluate_model(model, test_loader, device, criterion=None):
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
    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        #print("preds: ", preds)
        #print("labels:", labels)
        loss = criterion(outputs, labels).item()
        
        # statistics
        running_loss += loss * inputs.size(0)
        #print("preds == labels.data:", preds == labels.data)
        running_corrects += torch.sum(preds == labels.data).float()
        #print(running_corrects)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 50
reg = 1e-4
model.train()
model.to(device=device)
criterion = nn.CrossEntropyLoss()
graphs = {'train_loss':[], 'train_acc':[], 'test_loss':[], 'test_acc':[]}
for epoch in range(num_epochs):
    print("epoch:",epoch)
    st = time.time()
    running_loss = 0.0
    running_acc = 0.0
    for idx, item in enumerate(trainloader): 
        #print(item[0][0].shape, item[1][0])       
        inputs = item[0].to(device=device)
        #print(item[0][0].shape, item[1][0])
        labels = item[1].to(device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        #print("loss", loss)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
        #print("running_loss", running_loss)
        running_acc += (torch.sum(preds == labels.data).float() / preds.shape[0]).item()
        #print("running_accuracy", running_acc)
    
    graphs['train_loss'].append(running_loss)
    graphs['train_acc'].append(running_acc)
    test_loss, test_acc = evaluate_model(model, testloader, device, criterion)
    graphs['test_loss'].append(test_loss)
    graphs['test_acc'].append(test_acc)
    print("test loss: ", test_loss, "test acc: ", test_acc)

    # saving the trained model to a file
    torch.save(model.state_dict(), './gesture_xy.pt')
    log = "{} {}/{} loss:{:.4f} acc:{:.4f}\n".format("phase", idx, len(trainloader), running_loss / (idx+1), running_acc / (idx+1))
    print(log)
    print("time elapsed:", time.time()-st)

f = open("./gesture_graph_xy", "wb")
pickle.dump(graphs, f)

