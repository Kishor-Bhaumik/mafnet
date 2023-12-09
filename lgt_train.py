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
from model.kmodel_srm import MemNet
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
wandb_logger = WandbLogger(name= args.exp_name, project='MAFNet', log_model='all')


with open('config/config.yaml', 'r') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

now = datetime.datetime.now()
filename_log = 'Results-'+str(now)+'.txt'

# with torch.no_grad():
#     test_model = timm.create_model(cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
#     in_planes = test_model(torch.randn((2,3,128,128)))[0].shape[1]
#     del test_model

in_planes = 64

if cfg['dataset_params']['dataset_name'] == 'casia':
    from dataloader.loader import generator
elif cfg['dataset_params']['dataset_name'] == 'imd_2020':
    from dataloader.loader_imd import generator
elif cfg['dataset_params']['dataset_name'] == 'nist':
    from dataloader.loader_nist import generator
elif cfg['dataset_params']['dataset_name'] == 'coverage':
    from dataloader.loader_coverage import generator

gnr = generator(cfg)
training_generator = gnr.get_train_generator()
validation_generator = gnr.get_val_generator()

#model = MemNet(cfg, in_planes) #.to(device)



class lgt_model(pl.LightningModule):
    def __init__(self, Net, cfg ,in_planes):
        super().__init__()
        self.ourmodel= Net(cfg, in_planes)
        self.learning_rate = cfg['model_params']['lr']
        imbalance_weight = torch.tensor(cfg['dataset_params']['imbalance_weight']) #.to(device)
        self.criterion = nn.CrossEntropyLoss(weight = imbalance_weight)
        self.TrepletLoss_fake = torch.nn.TripletMarginLoss(margin=cfg['model_params']['triplet_margin_fake'])
        self.TrepletLoss_real = torch.nn.TripletMarginLoss(margin=cfg['model_params']['triplet_margin_real'])
        self.loss_mse = torch.nn.MSELoss()


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        img,tar = batch
        loss, mem_lossr, mem_lossf, auc, f1_macro, f1_micro = self.do_prediction(img,tar)

        self.log("training_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("training_mem_lossr", mem_lossr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("training_mem_lossf", mem_lossf, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("training_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("training_f1_macro",f1_macro, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("training_f1_micro",f1_micro, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
        
        # def lr_lambda(epoch):
        #     return 0.1 if epoch > 10 else 1.0  # Decrease learning rate by a factor of 10 after 10 epochs

        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        return self.optimizer

    # def on_epoch_end(self):
    #     if self.current_epoch > 10:
    #         for param_group in self.optimizer.param_groups:
    #             param_group['weight_decay'] = 0.01
    #     self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        img,tar = batch
        loss, mem_lossr, mem_lossf, auc, f1_macro, f1_micro = self.do_prediction(img,tar)
        self.log("val_mem_lossr", mem_lossr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_mem_lossf", mem_lossf, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, sync_dist=True,prog_bar=True , on_step=False, on_epoch=True,logger=True)
        self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_macro",f1_macro, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_f1_micro",f1_micro, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return auc
        #logits = self(x)
    
    def this_that_max(self,tar, y_score,matric,average='macro'):

        if matric == 'auc':
            for yy_true, yy_pred in zip(tar.detach().cpu().numpy(), y_score.detach().cpu().numpy()):
                this = metrics.roc_auc_score(yy_true.astype(int).ravel(), yy_pred.ravel())
                that = metrics.roc_auc_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel())
                auc = max(this, that)
            return auc

        else:
            threshold = 0.5
            for yy_true, yy_pred in zip(tar.detach().cpu().numpy(), y_score.detach().cpu().numpy()):
                yy_pred = (yy_pred > threshold).astype(int)
                this = metrics.f1_score(yy_true.astype(int).ravel(), yy_pred.ravel(),average=average)
                that = metrics.f1_score(yy_true.astype(int).ravel(), (1-yy_pred).ravel(),average=average)
                f1= max(this, that)
            return f1
    
    def do_prediction(self,img,tar):
        pred,  featr, memr, featf, memf= self.ourmodel(img,tar)

        mem_lossr = self.sum_gather_spread(featr,memr,self.TrepletLoss_real)
        mem_lossf = self.sum_gather_spread(featf, memf, self.TrepletLoss_fake)

        pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
        loss1 = self.criterion(pred, tar.long().detach()) 
        loss = 0.5*loss1+ 0.4*mem_lossr+ 0.1*mem_lossf
        #loss = loss1+ mem_lossr+ mem_lossf

        y_score = F.softmax(pred, dim=1)[:,1,:,:]
        auc = self.this_that_max(tar,y_score,matric='auc')
        f1_macro = self.this_that_max(tar,y_score,matric='f1', average= 'macro')
        f1_micro=  self.this_that_max(tar,y_score,matric='f1', average= 'micro')

        return loss, mem_lossr, mem_lossf, auc, f1_macro, f1_micro


    




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


# checkpoint_callback = ModelCheckpoint(
#     save_top_k=2,
#     monitor="val_auc",
#     mode="max",
#     dirpath="/",
#     filename="best_lgtmodel-{epoch:02d}-{val_auc:.2f}",
# )

checkpoint_callback = ModelCheckpoint(monitor="val_auc")

seed_everything(42, workers=True)
#Dmodel = LitDeepfake( dfmodel(), learning_rate=0.1, dataset= train_dataset )
# trainer =Trainer(max_epochs=5,callbacks=[checkpoint_callback])
# trainer.fit(Dmodel, train_loader, val_loader)

Dmodel = lgt_model( MemNet, cfg, in_planes)

# logger = TensorBoardLogger("tb_logs")
#trainer = Trainer(accelerator='gpu' , devices= [0,1,2,3,4] , strategy = 'ddp' ,max_epochs=100, callbacks=[checkpoint_callback],logger=wandb_logger)
trainer = Trainer(accelerator='gpu' , devices= [4],max_epochs=100, callbacks=[checkpoint_callback],
logger=wandb_logger,gradient_clip_val=0.5, gradient_clip_algorithm="value")
wandb_logger.watch(Dmodel)
trainer.fit(model=Dmodel, train_dataloaders=training_generator , val_dataloaders=validation_generator)
