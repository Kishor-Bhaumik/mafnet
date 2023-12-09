from tkinter import N
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import models
import warnings, torchmetrics
from pytorch_lightning.loggers import TensorBoardLogger
warnings.filterwarnings("ignore") 

import utils.utils as utils
from utils.utils import AverageMeter, batch_intersection_union, write_logger, set_random_seed
from sklearn import metrics
import timm
import yaml,pdb,statistics,os,datetime,wandb
from model.kmodel_sweep import MemNet
from torch.optim.lr_scheduler import StepLR
#from dataloader.loader import generator
from torch.optim.lr_scheduler import ReduceLROnPlateau
from argu import args
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

import logging
logging.getLogger("wandb").setLevel(logging.ERROR)
#'customized-encoder-theta-2layers_removed_detach_mem100'
#wandb.init(settings=wandb.Settings(console="off"))


'''
with open('config/config.yaml', 'r') as file:
    cfgg = yaml.load(file, Loader=yaml.FullLoader)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'


with torch.no_grad():
    test_model = timm.create_model(cfgg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
    in_planes = test_model(torch.randn((2,3,128,128)))[0].shape[1]
    del test_model

print(in_planes);exit()


if cfg['dataset_params']['dataset_name'] == 'casia':
    from dataloader.loader import generator
elif cfg['dataset_params']['dataset_name'] == 'imd_2020':
    from dataloader.loader_imd import generator
elif cfg['dataset_params']['dataset_name'] == 'nist':
    from dataloader.loader_nist import generator
elif cfg['dataset_params']['dataset_name'] == 'coverage':
    from dataloader.loader_coverage import generator
'''


in_planes = 2048

class lgt_model(pl.LightningModule):
    def __init__(self, model, cfg ,in_planes):
        super().__init__()
        self.ourmodel= model
        imbalance_weight = torch.tensor(cfg.imbalance_weight) #.to(device)
        self.criterion = nn.CrossEntropyLoss(weight = imbalance_weight)
        self.TrepletLoss_fake = torch.nn.TripletMarginLoss(margin=cfg.triplet_margin_fake)
        self.TrepletLoss_real = torch.nn.TripletMarginLoss(margin=cfg.triplet_margin_real)
        self.loss_mse = torch.nn.MSELoss()
        self.loss_wt_main= cfg.main_loss_weight
        self.loss_wt_fake = cfg.fake_mem_loss_weight
        self.loss_wt_real = cfg.real_mem_loss_weight
        self.cfg = cfg


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        img,tar = batch

        pred,  featr, memr, featf, memf= self.ourmodel(img,tar)

        mem_lossr = self.sum_gather_spread(featr,memr,self.TrepletLoss_real)
        mem_lossf = self.sum_gather_spread(featf, memf, self.TrepletLoss_fake)

        pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
        loss1 = self.criterion(pred, tar.long().detach()) 
        loss = self.loss_wt_main*loss1+ self.loss_wt_real*mem_lossr+ self.loss_wt_fake*mem_lossf
        #loss = loss1+ mem_lossr+ mem_lossf

        y_score = F.softmax(pred, dim=1)[:,1,:,:]

        for yy_true, yy_pred in zip(tar.detach().cpu().numpy(), y_score.detach().cpu().numpy()):
            this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
            that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel())
            auc=max(this, that)

        self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("training_mem_lossr", mem_lossr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("training_mem_lossf", mem_lossf, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("training_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.cfg.optimizer == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr = self.cfg.lr, weight_decay = 1e-4, momentum = 0.9)
        else:
            optimizer = optim.Adam(self.parameters(), lr = self.cfg.lr,  weight_decay=0.01)
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min'),
            'monitor': 'val_auc',  # Metric to monitor for lr scheduling
            'interval': 'epoch',
            'frequency': 1 }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def validation_step(self, batch, batch_idx):
        img,tar = batch

        pred,  featr, memr, featf, memf= self.ourmodel(img)

        mem_lossr = self.sum_gather_spread(featr,memr,self.TrepletLoss_real)
        mem_lossf = self.sum_gather_spread(featf, memf, self.TrepletLoss_fake)

        pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
        loss1 = self.criterion(pred, tar.long().detach()) 
        loss = self.loss_wt_main*loss1+ self.loss_wt_real*mem_lossr+ self.loss_wt_fake*mem_lossf

        y_score = F.softmax(pred, dim=1)[:,1,:,:]

        for yy_true, yy_pred in zip(tar.detach().cpu().numpy(), y_score.detach().cpu().numpy()):
            this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
            that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel())
            auc=max(this, that)

        self.log("val_auc", auc, sync_dist=True,prog_bar=True , on_step= False, on_epoch=True,logger=True)
        self.log("val_mem_lossr", mem_lossr, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_mem_lossf", mem_lossf, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, sync_dist=True,prog_bar=True , on_step=False, on_epoch=True,logger=True)

        return auc
        #logits = self(x)


    def spread_loss(self,query, keys, TrepletLoss):
        query= query.permute(0,2,3,1)
        batch_size, h,w,dims = query.size() # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)
        #1st, 2nd closest memories
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,1]]

        spreading_loss = TrepletLoss(query_reshape,pos.detach(), neg.detach())

        return spreading_loss
        
    def gather_loss(self, query, keys):
        query= query.permute(0,2,3,1)
        batch_size, h,w,dims = query.size() # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        gathering_loss = self.loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss
    
    def sum_gather_spread(self,feat,mem,trplt):

        gathering_loss = self.gather_loss(feat,mem)
        spreading_loss = self.spread_loss(feat, mem,trplt)

        return gathering_loss+spreading_loss

    def get_score(self, mem, query):
        bs,h,w,d= query.size()
        #pdb.set_trace()
        m, d = mem.size()
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)

        return score_query , score_memory


from dataloader.loader_nist_sweep import generator

def train():

    wandb.init(project="sweep_nist")
    config=wandb.config
    gnr = generator(config)
    wandb_logger = WandbLogger()

    training_generator = gnr.get_train_generator()
    validation_generator = gnr.get_val_generator()
    model = MemNet(config, in_planes)
   

    seed_everything(config.seed, workers=True)

    Dmodel = lgt_model( model, config, in_planes)

    # logger = TensorBoardLogger("tb_logs")
    #trainer = Trainer(accelerator='gpu' , devices= [0,1,2,3,4] , strategy = 'ddp' ,max_epochs=100, callbacks=[checkpoint_callback],logger=wandb_logger)
    trainer = Trainer(accelerator='gpu' , devices= [3] ,max_epochs=40,
    logger=wandb_logger,gradient_clip_val=0.5, gradient_clip_algorithm="value")
    trainer.fit(model=Dmodel, train_dataloaders=training_generator , val_dataloaders=validation_generator)


sweep_config = {
    'method': 'bayes',
    'metric': {'goal': 'maximize', 'name': 'val_auc'},
    'parameters': 
    {	
        # 'batch_size':  {'max': 64,
        #                 'min': 8,
        #                 'distribution': 'int_uniform'},
        'batch_size':  {'values':[8,16,32,64]},
        'lr':      {'max': 0.1,
                        'min': 0.0001,
                        'distribution': 'uniform'},
        'triplet_margin_fake':{'max': 1.0,
                        'min': 0.1,
                        'distribution': 'uniform'},
        'triplet_margin_real':{'max': 1.0,
                        'min': 0.1,
                        'distribution': 'uniform'},
        'seed':{'max': 1000,
                        'min':1 ,
                        'distribution': 'int_uniform'},
        'main_loss_weight' :{'max': 1.0,
                        'min': 0.1,
                        'distribution': 'uniform'},
        'fake_mem_loss_weight' :{'max': 1.0,
                        'min': 0.1,
                        'distribution': 'uniform'},
        'real_mem_loss_weight' :{'max': 1.0,
                        'min': 0.1,
                        'distribution': 'uniform'},
        'mem_dim' : {'max': 100,
                        'min': 10,
                        'distribution': 'int_uniform'},
        
        'optimizer': {'value': 'adam'},  
        'with_srm' : {'value' : True},
        'encoder': {'value': 'resnet50'},
        'fea_dim' :{'value': 64} ,
        'patch_size':{'value': 4},
        'aspp_outplane':{'value': 512} ,
        'imbalance_weight' :{'value':  [0.0892, 0.9108]},

        'dataset_name': {'value':'nist' }, #imd_2020 or casia or nist or coverage
        'base_dir': {'value': 'forgery/'},
        'images_dir_v1':{'value': '/home/data/forgery/CASIA1/CASIA1/Sp'} ,
        'mask_dir_v1':{'value': '/home/data/forgery/CASIA1/Gt/Sp'} ,
        'images_dir_v2':{'value': '/home/data/forgery/CASIA2/Tp' } ,
        'mask_dir_v2': {'value': '/home/data/forgery/CASIA2/Gt' },
        'imd_2020_dir': {'value': '/home/data/forgery/IMD2020' },
        'NIST_dir': {'value': '/home/data/forgery/NC2016_Test0613.SCI/NIST_images'},
        'coverage_dir': {'value': '/home/data/forgery/coverage/'},
        'im_size': {'value':224},
        'mean': {'value':  [0.485, 0.456, 0.406]} ,
        'std': {'value': [0.229, 0.224, 0.225]}

     }
}

# sweep_id = wandb.sweep(sweep_config, project="MAFNet")
# print("@@@@@@@@@@@@@@@@@@$$$$$$$$$$$$$$$$$$--------",sweep_id)
sweep_id = 'MAFNet/ctq4zf0h'
wandb.agent(sweep_id, function=train)
