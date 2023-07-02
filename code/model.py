import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

class AE_3D_Dataset(data.Dataset):
    def __init__(self, input, name='2d_cylinder', transform=None):
        # if name == 'SST':
        #     input = input[:,10:-10,20:-20]

        # NOTE 4: Prepare the hash map by craeting batches/ranges of 10-frames-patches
        self.input = input[:-10]
        self.target = input[10:]
        self.transform = transform
        # NOTE 4': each pth file represents 10 frames
        self.hashmap = {i:range(i, i+100, 10) for i in range(self.input.shape[0] - 100)}
        # self.hashmap = {i:range(i, i+10, 1) for i in range(self.input.shape[0] - 10)}
        print(len(self.hashmap))

    def __len__(self):
        return len(self.hashmap)

    def __getitem__(self, index):
        idx = self.hashmap[index]
        # print(idx)
        idy = self.hashmap[index]

        # NOTE 5: every 10s frame is obtained as a part of 10 frames/range thats read every iteration. Explain the indexing and draw it
        ip=self.input[idx]
        op=self.target[idy]

        # NOTE 6: batch dimentionality preparation
        x=self.transform(ip)
        x=x.permute(1, 2, 0)
        x=x.unsqueeze(0)

        y=self.transform(op)
        y=y.permute(1, 2, 0)
        y=y.unsqueeze(0)
        return x,y

############# UNet_3D ##########################################

class Downsample_3d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,stride,padding=(0,1,1)):
        super(Downsample_3d, self).__init__()
        self.net=nn.Sequential(
            nn.Conv3d(in_channel,in_channel,kernel_size=kernel,stride=stride,padding=padding,groups=in_channel),
            nn.Conv3d(in_channel,out_channel,1,1,0),
            nn.BatchNorm3d(out_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x=self.net(x)
        return x


class Upsample_3d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,stride,padding=(0,1,1)):
        super(Upsample_3d, self).__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose3d(in_channel,in_channel,kernel_size=kernel,stride=stride,padding=padding,groups=in_channel),
            nn.ConvTranspose3d(in_channel,out_channel,1,1,0),
            nn.BatchNorm3d(out_channel)
        )
        self.lRelu = nn.LeakyReLU()

    def forward(self,x1,x2,last=False):
        x=torch.cat((x1,x2),dim=1)
        x=self.net(x)
        if last:
            x=x
        else:
            x=self.lRelu(x)
        return x

def calculate_PCA(preds, dataset_name):
    p1_labels =  []
    p2_labels =  []
    p1_preds =  []
    p2_preds =  []

    # for img in range(labels.shape[0]):
    #     U, S, Vt = np.linalg.svd(labels[img])
    #     idx = np.argsort(S)[::-1]
    #     S = S[idx]
    #     U = U[:, idx]
    #     for i in range(U.shape[0]):
    #         p1_labels.append( U[i][0] * S[0] )
    #         p2_labels.append( U[i][1] * S[1] )
    U, S, Vt = np.linalg.svd(preds)
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    for i in range(U.shape[0]):
        p1_preds.append( U[i][0] * S[0] )
        p2_preds.append( U[i][1] * S[1] )
            
    
    # PCA
    # downscaled_labels = np.array([resize(img, (80, 100)) for img in labels])
    # labels_flattened = np.array([img.flatten() for img in downscaled_labels])
    # p1_labels, p2_labels = calculate_pca(labels_flattened)
    
    # downscaled_preds = np.array([resize(img, (80, 100)) for img in preds])
    # preds_flattened = np.array([img.flatten() for img in downscaled_preds])
    # p1_preds, p2_preds = calculate_pca(preds_flattened)

    import matplotlib.pyplot as plt
    # plt.plot(p1_labels, p2_labels, '-.', linewidth=0.1, color='blue', alpha = 0.5)
    plt.plot(p1_preds, p2_preds, '-.', linewidth=0.1, color='red', alpha=0.5)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Temporal Image Data')
    plt.savefig(
            f"../results/{dataset_name}/p1p2_pca.png",
            dpi=600,
            bbox_inches="tight",
            pad_inches=0)
    plt.close()

class UNet_3D(nn.Module):
    def __init__(self,name):
        super(UNet_3D, self).__init__()
        self.name=name

        if name=='2d_cylinder_CFD' or name=='2d_cylinder' or name=='2d_sq_cyl':
            d1=Downsample_3d(1, 16, (3, 3, 8), stride=(1, 1, 4), padding=(0, 1, 2)) #16,80,80
            u5=Upsample_3d(32, 1, (3, 3, 8), stride=(1, 1, 4), padding=(0, 1, 2)) #190,360

        elif name=='2d_airfoil':
            d1=Downsample_3d(1, 16, (3, 3, 4), stride=(1, 1, 2), padding=(0, 1, 1)) #16,80,80
            u5=Upsample_3d(32, 1, (3, 3, 4), stride=(1, 1, 2), padding=(0, 1, 1)) #190,360

        elif name=='boussinesq':
            d1= Downsample_3d(1,16,(3,8,4),stride=(1,4,2),padding=(0,2,1))
            u5 = Upsample_3d(32,1,(3,8,4),stride=(1,4,2),padding=(0,2,1))
            # u6 = nn.ConvTranspose3d(8,1,(3,6,3),stride=(1,3,1),padding=(1,0,1))
            # self.u6=u6

        elif name=='SST' or name=='2d_plate':
            #Note - Remember to crop in dataloader
            d1=Downsample_3d(1, 16, (3, 4, 8), stride=(1, 2, 4), padding=(0, 1, 2))
            u5=Upsample_3d(32,1,(3,4,8),stride=(1,2,4),padding=(0,1,2))

        elif name=='channel_flow':
            d1=Downsample_3d(1, 16, (3, 8, 3), stride=(1, 4, 1), padding=(0, 2, 1)) #16,80,80
            u5=Upsample_3d(32, 1, (3, 8, 3), stride=(1, 4, 1), padding=(0, 2, 1)) #190,360

        else:
            print(f'Dataset Not Defined')

        # NOTE 3: Define Down and Up sampling dimensions, here we have to ADAPT input and OUTPUT only -> intrusive step
        self.d1=d1
        self.d2=Downsample_3d(16 , 32 , (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #44
        self.d3=Downsample_3d(32 , 64 , (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #22
        self.d4=Downsample_3d(64 , 128, (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #10
        self.d5=Downsample_3d(128, 256, (2, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #5

        self.h = 32
        self.down = nn.Linear(256*5*5, self.h)
        self.up = nn.Linear(self.h, 256*5*5)

        self.u1=Upsample_3d(512, 128, (2, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #10
        self.u2=Upsample_3d(256, 64 , (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #22
        self.u3=Upsample_3d(128, 32 , (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #44
        self.u4=Upsample_3d(64 , 16 , (3, 4, 4) ,stride=(1, 2, 2), padding=(0, 1, 1)) #90
        self.u5=u5#190,360
        
        # self.global_preds = np.empty((0,6400))


    def forward(self,x):

        down1=self.d1(x)
        down2=self.d2(down1)
        down3=self.d3(down2)
        down4=self.d4(down3)
        down5=self.d5(down4)

        conv_shape = down5.shape
        
        mid = down5.view(down5.shape[0], -1)
        # ####
        # if mid.is_cuda:
        #     mid2 = mid.cpu() 
        # mid2 = mid2.detach().numpy()
        # self.global_preds = np.vstack(( self.global_preds, mid2 ))
        
        # if self.global_preds.shape[0] == 501:
        #     calculate_PCA(self.global_preds, self.name)
        # print(self.global_preds.shape)
        #####
        mid = self.down(mid)
        mid= F.relu(mid)
        
        mid = self.up(mid)
        mid= F.relu(mid)
        mid = mid.view(mid.shape)
        mid = mid.view(conv_shape)

        up1=self.u1(down5,mid)
        up2=self.u2(down4,up1)
        up3=self.u3(down3,up2)
        up4=self.u4(down2,up3)

        # if self.name=='boussinesq':
        #     out= self.u5(down1,up4)
        #     out =self.u6(out)

        # else:
        out = self.u5(down1,up4,last=True)

        return out
