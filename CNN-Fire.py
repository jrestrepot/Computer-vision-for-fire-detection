from matplotlib import pyplot as plt
import numpy as np
import torchvision
import numpy as np
import torch 
import torch.nn as nn
from torch.nn.functional import relu
from torch.optim import SGD
from torchinfo import summary
import torchvision.transforms as T
from torchvision.transforms import ToTensor, Compose, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
from torch.utils.data import Subset

dataset_path = '/workspaces/Nueva carpeta/content/folder'
classes = ('Non fire', 'Fire')

training_transforms = Compose([transforms.Resize((300, 300)), transforms.ToTensor()])
train_dataset = torchvision.datasets.ImageFolder(root= dataset_path, transform= training_transforms)
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= 30, shuffle= False)

#Mean and Standard Deviation of each image
def get_mean_std(loader):
  mean = 0
  std = 0
  total_images = 0
  for images, _ in loader:
    images_in_a_batch = images.size(0)
    images = images.view(images_in_a_batch, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_images += images_in_a_batch
  
  mean /= total_images
  std /= total_images

  return mean,std

mean, std = get_mean_std(train_loader)

#Images preprocessing
transforms2 = Compose(
        [T.Resize((300,300)),
         T.RandomHorizontalFlip(),
         ToTensor(),
         T.Normalize(mean, std)
         ]) 

imageFolder = torchvision.datasets.ImageFolder(root= dataset_path, transform= transforms2)

def train_val_dataset(dataset, val_split):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

first_split = train_val_dataset(imageFolder, 0.3)

final_testing_loader = torch.utils.data.DataLoader(first_split['val'], batch_size= 30, shuffle= True)

datasets = train_val_dataset(first_split['train'], 0.25)

dataloaders = {x:torch.utils.data.DataLoader(datasets[x], batch_size= 30, shuffle=True) for x in ['train','val']}
train_dataloader = dataloaders['train']
test_dataloader = dataloaders['val']

# Function that shows random images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Random images from dataset
dataiter = iter(train_dataloader)
images, labels = dataiter.next()


# Show images
imshow(torchvision.utils.make_grid(images))

#Construction of the neural network
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,5,5) #Tunear filtros o kernel
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(5,10,5) #Tunear filtros o kernel
        self.fc1 = nn.Linear(72*72*10,100) 
        self.fc2 = nn.Linear(100,10) #Tunear batch size
        self.fc3 = nn.Linear(10,1)

    def forward(self, x):
        x = self.pool(relu(self.conv1(x)))
        x = self.pool(relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

net = MyNet()
criterion = nn.BCELoss()
optimizer = SGD(net.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(15): #Tunear esto
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'[{epoch + 1}] loss: {running_loss/25:.3f}')

dataiter = iter(test_dataloader)
images, labels = dataiter.next()

#Print images
imshow(torchvision.utils.make_grid(images))

print('Real Labels: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(30)))

outputs = net(images)

results = outputs.tolist()

predictions = []
for i in results:
    if i > 0.5:
        predictions.append(1)
    else:
        predictions.append(0)

cont = 0
lab = labels.tolist()
for i in range(30):
    if lab[i] == predictions[i]:
        cont += 1
cont /= 30
print(cont)

#Partial validation of the model (Validation set)
correct = 0
total = 0
true = []
pred = []
full_outs = []

with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images)
        outputs = outputs.tolist()
        full_outs += outputs

        predictions = list()
        for i in outputs:
            if i > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        total += labels.size(0)
        lab = labels.tolist()
        true += lab
        pred += predictions 
        
        for i in range(labels.size(0)):
            if lab[i] == predictions[i]:
                correct += 1
                
#Evaluation metrics:
print(f"Accuracy of the network is: {correct / total}")

#Graph confusion matrix
if len(true) != len(pred):
    print(len(true))
    print(len(pred))
else:
    cf = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf/len(true), index = [j for j in classes], columns = [j for j in classes])
    plt.figure(figsize = (12, 7))
    sns.heatmap(df_cm, annot = True)
    plt.savefig('Confusion_matrix956')

#Recall represents True Positive Rate
recall = cf[1][1]/(cf[1][1] + cf[1][0])

#Precision represents False Postitve Rate
precision = cf[1][1]/(cf[1][1] + cf[0][1])

print("Exhaustividad del Modelo: {} y Precisión del Modelo: {}".format(recall.round(5), precision.round(5)))

#F1 represents armonic mean of recall and precision
f1 = f1_score(true, pred)

print(f"F1-score del Modelo: {f1}")

fpr, tpr, _ = roc_curve(true, full_outs)

# Graph ROC Curve
plt.plot(fpr, tpr)
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()
plt.savefig('CurvaROC')

#AUC Score
auc_score = roc_auc_score(true, full_outs)

print(f"ROC-AUC del Modelo: {auc_score}")

#Total validation of the model (Test set):
correct = 0
total = 0
true = []
pred = []
full_outs = []

with torch.no_grad():
    for data in final_testing_loader:
        images, labels = data
        outputs = net(images)
        outputs = outputs.tolist()
        full_outs += outputs

        predictions = list()
        for i in outputs:
            if i > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        total += labels.size(0)
        lab = labels.tolist()
        true += lab
        pred += predictions 
        
        for i in range(labels.size(0)):
            if lab[i] == predictions[i]:
                correct += 1
                
#Evaluation metrics:
print(f"Accuracy of the network is: {correct / total}")

#Graph confusion matrix
if len(true) != len(pred):
    print(len(true))
    print(len(pred))
else:
    cf = confusion_matrix(true, pred)
    df_cm = pd.DataFrame(cf/len(true), index = [j for j in classes], columns = [j for j in classes])
    plt.figure(figsize = (12, 7))
    sns.heatmap(df_cm, annot = True)
    plt.savefig('Confusion_matrix956')

#Recall represents True Positive Rate
recall = cf[1][1]/(cf[1][1] + cf[1][0])

#Precision represents False Postitve Rate
precision = cf[1][1]/(cf[1][1] + cf[0][1])

print("Exhaustividad del Modelo: {} y Precisión del Modelo: {}".format(recall.round(5), precision.round(5)))

#F1 represents armonic mean of recall and precision
f1 = f1_score(true, pred)

print(f"F1-score del Modelo: {f1}")

fpr, tpr, _ = roc_curve(true, full_outs)

# Graph ROC Curve
plt.plot(fpr, tpr)
plt.xlabel('True Positive Rate')
plt.ylabel('False Positive Rate')
plt.show()
plt.savefig('CurvaROC')

#AUC Score
auc_score = roc_auc_score(true, full_outs)

print(f"ROC-AUC del Modelo: {auc_score}")
