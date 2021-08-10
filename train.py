import torch
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
import imageio
import itertools
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pathlib import Path

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Path("./results/samples").mkdir(parents=True, exist_ok=True)

transforms1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def image_prepare(img,size):
    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]
    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]
    img = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_NEAREST)
    if (np.ndim(img) == 3):
        out_img = img
    else:
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img    
    if np.max(out_img)>1:
        out_img = out_img/255.0
    return out_img

# Image Dataset
class Img_Dataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_idx = self.images[idx]
        img_idx = imageio.imread(self.images[idx])
        img_idx = image_prepare(img_idx, 127)
        img_nrml = transforms1(img_idx)
        del img_idx
        sample = {'image': img_nrml}
        return sample

# Residual Block for the Encoder
class ResBlock(torch.nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256,256,3,1,1, bias=False),
            torch.nn.Conv2d(256,256,3,1,1, bias=False),
            )
        self.LeakyReLU = torch.nn.LeakyReLU(0.2)
    def forward(self, x):
        return self.LeakyReLU(self.block(x) + x)

# Inverted Residual Block for the Decoder
class InvResBlock(torch.nn.Module):
    def __init__(self):
        super(InvResBlock, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256,256,3,1,1, bias=False),
            torch.nn.Conv2d(256,256,3,1,1, bias=False),
            )
        self.LeakyReLU = torch.nn.LeakyReLU(1/0.2)
    def forward(self, x):
        return (self.block(self.LeakyReLU(x)) + self.LeakyReLU(x))


class Transformer(torch.nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.genB = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,7,1,3),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64,128,3,2),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128,256,3,2),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),            
            torch.nn.ConvTranspose2d(256,128,3,2),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128,64,3,2),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64,3,7,1,3),
            torch.nn.Tanh()
        )
        self.genA = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,7,1,3),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64,128,3,2),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128,256,3,2),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            InvResBlock(),
            torch.nn.ConvTranspose2d(256,128,3,2),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128,64,3,2),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64,3,7,1,3),
            torch.nn.Tanh()
        )
        self.discA = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,4,2),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64,128,4,2),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128,256,4,2),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256,512,4,1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 512, 4, 1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1, 8, 1),
            torch.nn.Sigmoid()
        )
        self.discB = torch.nn.Sequential(
            torch.nn.Conv2d(3,64,4,2),
            torch.nn.InstanceNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64,128,4,2),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128,256,4,2),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256,512,4,1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 512, 4, 1),
            torch.nn.InstanceNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1, 8, 1),
            torch.nn.Sigmoid()
        )
    def forward(self, x_a,x_b):
        y_a = self.genA(x_b)
        y_b = self.genB(x_a)

        a = self.discA(x_a)
        b = self.discA(y_a)
        c = self.discB(x_b)
        d = self.discB(y_b)
        e = self.genA(y_b)
        f = self.genB(y_a)
        
        return a,b,c,d,e,f

# List of images names
day_image_names_list = os.listdir('./data/day/')
night_image_names_list = os.listdir('./data/night/')

# List of images paths
day_image_names_list = ['./data/day/'+x for x in day_image_names_list]
night_image_names_list = ['./data/night/'+x for x in night_image_names_list]

day_images_dataset = Img_Dataset(day_image_names_list)
night_images_dataset = Img_Dataset(night_image_names_list)

batch_size = 16
day_images_dataloader = DataLoader(day_images_dataset, batch_size = batch_size, shuffle=True, num_workers = 0, drop_last = True)
night_images_dataloader = DataLoader(night_images_dataset, batch_size = batch_size, shuffle=True, num_workers = 0, drop_last = True)

transformer = Transformer().float().to(device)

# Default hyperparameters
lr = 0.0002
optimizer = torch.optim.Adam(transformer.parameters(), lr=lr, betas=(0.5, 0.999))

l1_loss = torch.nn.L1Loss()
adversarial_loss = torch.nn.BCELoss()

epochs = 1000

test_day_image = next(iter(day_images_dataloader))['image'][0].float().to(device)
test_night_image = next(iter(night_images_dataloader))['image'][0].float().to(device)

for epoch in range(epochs):
    
    epoch_loss = 0    

    for batch_num, (day_data, night_data) in enumerate(zip((day_images_dataloader), itertools.cycle(night_images_dataloader))):

        print('\r{}'.format(str(int(batch_num))+'/'+str(int(8035/batch_size))), end = '\r')    
        
        day_images = day_data['image'].float().to(device)
        night_images = night_data['image'].float().to(device)

        a,b,c,d,e,f = transformer(day_images,night_images)
        
        L_adv_A = adversarial_loss(a,torch.ones(a.size()).to(device)) + adversarial_loss(b,torch.zeros(b.size()).to(device))
        L_adv_B = adversarial_loss(c,torch.ones(c.size()).to(device)) + adversarial_loss(d,torch.zeros(d.size()).to(device))
        Lcc = l1_loss(day_images,e) + l1_loss(night_images,f)

        total_loss = L_adv_A + L_adv_B + 10*Lcc

        epoch_loss += total_loss.item()

        transformer.zero_grad()
        total_loss.backward()
        optimizer.step()

    print('\nEpoch: '+str(epoch)+'/'+str(epochs)+' Loss: '+str(epoch_loss/batch_num)+'\n' )

    fig, axs = plt.subplots(nrows=4,ncols=2, sharex=True, figsize=(16, 32))    
    axs[0][0].set_title('Day Original', fontsize=40)
    axs[0][0].imshow((test_day_image.detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[0][1].set_title('Day Reconstructed', fontsize=40)
    axs[0][1].imshow((transformer.genA(transformer.genB(test_day_image.unsqueeze(0)))[0].detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[1][0].set_title('Night Original', fontsize=40)
    axs[1][0].imshow((test_night_image.detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[1][1].set_title('Night Reconstructed', fontsize=40)
    axs[1][1].imshow((transformer.genB(transformer.genA(test_night_image.unsqueeze(0)))[0].detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[2][0].set_title('Day Original', fontsize=40)
    axs[2][0].imshow((test_day_image.detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[2][1].set_title('Day to Night', fontsize=40)
    axs[2][1].imshow((transformer.genB(test_day_image.unsqueeze(0))[0].detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[3][0].set_title('Night Original', fontsize=40)
    axs[3][0].imshow((test_night_image.detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')

    axs[3][1].set_title('Night to Day', fontsize=40)
    axs[3][1].imshow((transformer.genA(test_night_image.unsqueeze(0))[0].detach().cpu().permute(1,2,0).numpy()+1)/2, origin='upper')
    
    fig.savefig('./results/samples/epoch'+str(epoch)+'.png', dpi=300)
