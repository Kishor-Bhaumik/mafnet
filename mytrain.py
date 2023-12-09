
#trainer.py..
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from tqdm import tqdm
import utils.utils as utils
import torch.optim as optim
from utils.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
from sklearn import metrics
import datetime
import timm
import yaml,pdb,statistics,os
from model.kmodel import CFLNet
from torch.optim.lr_scheduler import StepLR

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning) 
os.environ["LIBPNG_WARNINGS"] = "0"

set_random_seed(1221)

use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")

with open('config/config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

with torch.no_grad():
    test_model = timm.create_model(cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
    in_planes = test_model(torch.randn((2,3,128,128)))[0].shape[1]
    del test_model



from dataloader.loader import generator


gnr = generator(cfg)
training_generator = gnr.get_train_generator()
validation_generator = gnr.get_val_generator()

model = CFLNet(cfg, in_planes).to(device)



if cfg['model_params']['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), 
            lr = cfg['model_params']['lr'], weight_decay = 1e-4, momentum = 0.9)
else:
    optimizer = optim.Adam(model.parameters(), 
            lr = cfg['model_params']['lr'])

scheduler = StepLR(optimizer, step_size=20, gamma=0.8)




imbalance_weight = torch.tensor(cfg['dataset_params']['imbalance_weight']).to(device)
criterion = nn.CrossEntropyLoss(weight = imbalance_weight)




# from torch.utils.tensorboard import SummaryWriter

# tb = SummaryWriter()

# inp = torch.randn(4,3,224,224).to(device)
# inp = torch.tensor(inp, dtype= torch.float)
# tar = torch.randint( 0,2 ,(4,224,224)).to(device)
# tar= torch.tensor(tar, dtype= torch.float)
# pred, mem_lossr, mem_lossf = model(inp,tar)



def main():
    max_val_auc = 0
    max_val_iou = [0.0, 0.0]
    for epoch in range(cfg['model_params']['epoch']):
        train_loss = AverageMeter()
        train_sloss = AverageMeter()
        train_closs = AverageMeter()

        loss_1 = []
        lossr = []
        lossf = []
        
        for sample in training_generator: # tqdm(training_generator):
            model.train()
            optimizer.zero_grad()

            img = sample[0].to(device)
            tar = sample[1].to(device)

            pred, mem_lossr, mem_lossf= model(img,tar)
            
            pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)

            loss1 = criterion(pred, tar.long().detach()) 

            total_loss = 0.5*loss1+ 0.3*mem_lossr+ 0.2*mem_lossf
            loss_1.append(loss1.detach().cpu().item())
            lossr.append(mem_lossr.detach().cpu().item())
            lossf.append(mem_lossf.detach().cpu().item())
            
            total_loss.backward()

            optimizer.step()

        print("epochs: ", epoch, statistics.mean(loss_1), statistics.mean(lossf), statistics.mean(lossr))
        
        scheduler.step()    
            
        # train_softmax = train_sloss.avg

        # if cfg['global_params']['with_con'] == True:
        #     train_contrast = train_closs.avg
        
        
        # with torch.no_grad():
        #     model.eval()

        #     val_pred = []
        #     val_tar = []
        #     auc = []
        #     for img, tar in tqdm(validation_generator):
        #         img, tar = img.to(device), tar.to(device)
        #         pred, _ = model(img)
        #         pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
            
                
        #         y_score = F.softmax(pred, dim=1)[:,1,:,:]
                
        #         # the following auc code is taken from:
        #         # https://github.com/ZhiHanZ/IRIS0-SPAN/blob/main/utils/metrics.py
                
        #         for yy_true, yy_pred in zip(tar.cpu().numpy(), y_score.cpu().numpy()):
        #             this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
        #             that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel())
        #             auc.append(max(this, that))
                

        #     val_auc = np.mean(auc)

        #     val_pred = []
        #     val_tar = []

        #     if val_auc > max_val_auc:
        #         max_val_auc = val_auc


        #     if cfg['global_params']['with_con'] == True:
        #         logs = {'epoch': epoch, 'Softmax Loss':train_softmax, 'Contrastive Loss':train_contrast, 
        #         'Validation AUC': val_auc, 'Max Validaton_AUC': max_val_auc}
            
        #     else:
        #         logs = {'epoch': epoch, 'Softmax Loss':train_softmax,
        #         'Validation AUC': val_auc, 'Max Validaton_AUC': max_val_auc}

        #     # tb.add_scalar("auc", val_auc, epoch+1)
        #     write_logger(filename_log, cfg, **logs)


if __name__ == '__main__':

    main()


        
        
           
