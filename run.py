import random

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import LodgedBarleyDataset, LodgedBarleyDatasetForTest
from transform import preprocessing


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    ds_test = LodgedBarleyDatasetForTest(root='data/', transform=preprocessing)
    
    loc = 'save_model/FPN_preprocess_1_4'
    model = torch.load(loc +'.pt')
    model = model.cuda()
    model.eval()

    pick = []
    for i in range(1):
        pick.append(random.randrange(0, 30, 1))

    for i in pick:
        X, y = ds_test.__getitem__(i)
        # torchvision.utils.save_image(X, './testimage/'+loc.split('/')[-1]+'_'+str(i)+'_X'+'.png')
        # torchvision.utils.save_image(y, './testimage/'+loc.split('/')[-1]+'_'+str(i)+'_y'+'.png')
        print(X.size())
        torchvision.utils.save_image(X[0:3, :, :], './testimage/'+loc.split('/')[-1]+'_'+"target"+'_X'+'.png')
        torchvision.utils.save_image(y, './testimage/'+loc.split('/')[-1]+'_'+"target"+'_y'+'.png')
        X = X.view(1, 4, 512, 512).cuda()
        y_pred = model(X)
        print(y_pred)
        torchvision.utils.save_image(y_pred, './testimage/'+loc.split('/')[-1]+'_'+"target"+'_ypred'+'.png')