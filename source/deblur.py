# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import glob
# import cv2
# import torch
# import torchvision
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import time
# import albumentations
# import argparse
# import models

# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
# from torchvision.utils import save_image
# from sklearn.model_selection import train_test_split

# # constructing the argument parser
# parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--epochs', type=int, default=50, 
#             help='number of epochs to train the model for')
# args = vars(parser.parse_args())

# # helper functions
# image_dir = 'outputs/saved_images'
# os.makedirs(image_dir, exist_ok=True)
    
# def save_decoded_image(img, name):
#     img = img.view(img.size(0), 3, 224, 224)
#     save_image(img, name)

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# print(device)

# batch_size = 2

# gauss_blur = os.listdir('number_plates_image/blur_images')
# gauss_blur.sort()
# sharp = os.listdir('number_plates_image/clear_images')
# sharp.sort()

# x_blur = []
# for i in range(len(gauss_blur)):
#     x_blur.append(gauss_blur[i])

# y_sharp = []
# for i in range(len(sharp)):
#     y_sharp.append(sharp[i])
    
# print(x_blur[10])
# print(y_sharp[10])

# (x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)

# print(len(x_train))
# print(len(x_val))

# # define transforms
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# class DeblurDataset(Dataset):
#     def __init__(self, blur_paths, sharp_paths=None, transforms=None):
#         self.X = blur_paths
#         self.y = sharp_paths
#         self.transforms = transforms
         
#     def __len__(self):
#         return (len(self.X))
    
#     def __getitem__(self, i):
#         blur_image = cv2.imread(f"number_plates_image/blur_images/{self.X[i]}")
        
#         if self.transforms:
#             blur_image = self.transforms(blur_image)
            
#         if self.y is not None:
#             sharp_image = cv2.imread(f"number_plates_image/clear_images/{self.y[i]}")
#             sharp_image = self.transforms(sharp_image)
#             return (blur_image, sharp_image)
#         else:
#             return blur_image

# train_data = DeblurDataset(x_train, y_train, transform)
# val_data = DeblurDataset(x_val, y_val, transform)
 
# trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# model = models.DeepCNN().to(device)
# print(model)

# # the loss function
# criterion = nn.MSELoss()
# # the optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
#         optimizer,
#         mode='min',
#         patience=5,
#         factor=0.1,
#         # verbose=True
#     )

# def fit(model, dataloader, epoch):
#     model.train()
#     running_loss = 0.0
#     for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
#         blur_image = data[0]
#         sharp_image = data[1]
#         blur_image = blur_image.to(device)
#         sharp_image = sharp_image.to(device)
#         optimizer.zero_grad()
#         outputs = model(blur_image)
#         loss = criterion(outputs, sharp_image)
#         # backpropagation
#         loss.backward()
#         # update the parameters
#         optimizer.step()
#         running_loss += loss.item()
    
#     train_loss = running_loss/len(dataloader.dataset)
#     print(f"Train Loss: {train_loss:.5f}")
    
#     return train_loss

# # the training function
# def validate(model, dataloader, epoch):
#     model.eval()
#     running_loss = 0.0
#     with torch.no_grad():
#         for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
#             blur_image = data[0]
#             sharp_image = data[1]
#             blur_image = blur_image.to(device)
#             sharp_image = sharp_image.to(device)
#             outputs = model(blur_image)
#             loss = criterion(outputs, sharp_image)
#             running_loss += loss.item()

#             if epoch == 0 and i == (len(val_data)/dataloader.batch_size)-1:
#                 save_decoded_image(sharp_image.cpu().data, name=f"outputs/saved_images/sharp{epoch}.jpg")
#                 save_decoded_image(blur_image.cpu().data, name=f"outputs/saved_images/blur{epoch}.jpg")

#         val_loss = running_loss/len(dataloader.dataset)
#         print(f"Val Loss: {val_loss:.5f}")

#         save_decoded_image(outputs.cpu().data, name=f"outputs/saved_images/val_deblurred{epoch}.jpg")
        
#         return val_loss

# train_loss  = []
# val_loss = []
# start = time.time()
# for epoch in range(args['epochs']):
#     print(f"Epoch {epoch+1} of {args['epochs']}")
#     train_epoch_loss = fit(model, trainloader, epoch)
#     val_epoch_loss = validate(model, valloader, epoch)
#     train_loss.append(train_epoch_loss)
#     val_loss.append(val_epoch_loss)
#     scheduler.step(val_epoch_loss)
# end = time.time()

# print(f"Took {((start-end)/60):.3f} minutes to train")

# # loss plots
# plt.figure(figsize=(10, 7))
# plt.plot(train_loss, color='orange', label='train loss')
# plt.plot(val_loss, color='red', label='validataion loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('outputs/loss.png')
# plt.show()

# # save the model to disk
# print('Saving model...')
# torch.save(model.state_dict(), 'outputs/model.pth')


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import models

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs to train the model for')
args = vars(parser.parse_args())

image_dir = 'outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)

def save_decoded_image(img, name):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.dim() == 4:
            img = img.view(img.size(0), 3, 224, 224)
        img = torch.clamp(img, 0.0, 1.0)
    save_image(img, name)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 2

gauss_blur = os.listdir('number_plates_image/blur_images')
gauss_blur.sort()
sharp = os.listdir('number_plates_image/clear_images')
sharp.sort()

x_blur = [f for f in gauss_blur]
y_sharp = [f for f in sharp]

if len(x_blur) > 10:
    print(x_blur[10])
if len(y_sharp) > 10:
    print(y_sharp[10])

(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.25)

print(len(x_train))
print(len(x_val))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        blur_image = cv2.imread(f"number_plates_image/blur_images/{self.X[i]}")
        if blur_image is None:
            raise FileNotFoundError(f"Couldn't read blur image: number_plates_image/blur_images/{self.X[i]}")
        blur_image = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            blur_image = self.transforms(blur_image)
        if self.y is not None:
            sharp_image = cv2.imread(f"number_plates_image/clear_images/{self.y[i]}")
            if sharp_image is None:
                raise FileNotFoundError(f"Couldn't read sharp image: number_plates_image/clear_images/{self.y[i]}")
            sharp_image = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = models.DeepCNN().to(device)
print(model)

# verify model parameters exist (debug)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.1,
    )

def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size) if dataloader.batch_size else len(dataloader)):
        blur_image, sharp_image = data
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    print(f"Train Loss: {train_loss:.5f}")
    return train_loss

def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader), 0):
            blur_image, sharp_image = data
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()
            if epoch == 0 and i == len(dataloader)-1:
                save_decoded_image(sharp_image.cpu().data, name=f"outputs/saved_images/sharp{epoch}.jpg")
                save_decoded_image(blur_image.cpu().data, name=f"outputs/saved_images/blur{epoch}.jpg")
        val_loss = running_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        print(f"Val Loss: {val_loss:.5f}")
        save_decoded_image(torch.clamp(outputs.cpu().data, 0.0, 1.0), name=f"outputs/saved_images/val_deblurred{epoch}.jpg")
        return val_loss

train_loss  = []
val_loss = []
start = time.time()
for epoch in range(args['epochs']):
    print(f"Epoch {epoch+1} of {args['epochs']}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()

print(f"Took {((end-start)/60):.3f} minutes to train")

plt.figure(figsize=(10, 7))
plt.plot(train_loss, label='train loss')
plt.plot(val_loss, label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/loss.png')
plt.show()

print('Saving model...')
os.makedirs('outputs', exist_ok=True)
torch.save(model.state_dict(), 'outputs/model_deepcnn.pth')
