

import torch,pdb
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from utils.utils import compute_IoU, FScore,  get_confusion_matrix, IOU_loss
from sklearn import metrics

class lgt_model(pl.LightningModule):
    def __init__(self, Net, cfg , dataloader_length):
        super(lgt_model, self).__init__()
        self.ourmodel= Net
        self.criterion = nn.CrossEntropyLoss(weight = torch.tensor(cfg.imbalance_weight))
        self.TrepletLoss_fake = torch.nn.TripletMarginLoss(margin=cfg.triplet_margin_fake)
        self.TrepletLoss_real = torch.nn.TripletMarginLoss(margin=cfg.triplet_margin_real)
        self.loss_mse = torch.nn.MSELoss()
        self.len_dataloader = dataloader_length
        self.cfg = cfg
        self.iou_loss =IOU_loss()
        self.bce = nn.BCELoss(weight = torch.tensor(cfg.imbalance_weight))
        self.highest_auc = 0
        self.highest_Fscore = 0
        self.highest_Fscores = []

        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        img,tar = batch
        #loss, mem_lossr, mem_lossf, auc, f1_macro, f1_micro,_,_ = self.do_prediction(img,tar)

        if self.cfg.memory_module:
            pred,  featr, memr, featf, memf, attn_map_r, attn_map_f= self.ourmodel(img,tar)
            loss,_,_,_,_,_,_,_,_= self.do_prediction(pred,tar,img,  featr, memr, featf, memf, attn_map_r, attn_map_f)
            self.log("training_total_loss", loss[0], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("training_crosEntr_loss", loss[1], on_step=False, on_epoch=True, prog_bar=False, logger=True)
            return loss[0]
        else:
            pred= self.ourmodel(img,tar)
            loss,_,_,_,_,_,_= self.do_prediction(pred,tar,img)
            return loss

        # self.log("training_mem_lossr", mem_lossr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log("training_mem_lossf", mem_lossf, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log("training_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log("training_f1_macro",f1_macro, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log("training_f1_micro",f1_micro, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    # def configure_optimizers(self):
    #     self.optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
    #     return self.optimizer

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        optimizer = optim.SGD([{'params':
                                    filter(lambda p: p.requires_grad,
                                            self.parameters()),
                                'lr': self.cfg.lr}],
                            lr=self.cfg.lr,
                            momentum=self.cfg.momentum,
                            weight_decay=self.cfg.weight_decay,
                            nesterov=False,
                            )

        def lr_lambda(current_step):
            max_iters = self.trainer.max_epochs * self.len_dataloader
            return (1 - current_step / max_iters) ** self.cfg.power

        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',  # or 'epoch' for epoch-level scheduling
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


    def validation_step(self, batch, batch_idx):

        if self.trainer.sanity_checking:
            return

        img,tar = batch

        if self.cfg.memory_module:
            pred,  featr, memr, featf, memf, attn_map_r, attn_map_f= self.ourmodel(img)
            loss, mem_lossr, mem_lossf, auc, f1_macro,  avg_p_mIoU, m_f1,iou_harmonious, Fscore_harmonious = self.do_prediction(pred,tar,img,  featr, memr, featf, memf, attn_map_r, attn_map_f)  
            self.highest_Fscores.append(Fscore_harmonious)
            self.log("Fscore_harmonious",Fscore_harmonious, on_step=False, on_epoch=True, prog_bar=True, logger=True)                 
            self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("val_f1_macro",f1_macro , on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_mem_lossr", mem_lossr, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_mem_lossf", mem_lossf, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_total_loss", loss[0], sync_dist=True,prog_bar=False , on_step=False, on_epoch=True,logger=True)
            self.log("val_crossEntr_loss", loss[1], sync_dist=True,prog_bar=False , on_step=False, on_epoch=True,logger=True)

            return {"Fscore_harmonious" : Fscore_harmonious}

        else:
            pred= self.ourmodel(img)
            loss,auc, f1_macro, avg_p_mIoU, m_f1,iou_harmonious, Fscore_harmonious = self.do_prediction(pred,tar,img)
            self.log("Fscore_harmonious",Fscore_harmonious, on_step=False, on_epoch=True, prog_bar=True, logger=True)        
            self.log("val_auc", auc, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_f1_macro",f1_macro , on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_crossEntr_loss", loss, sync_dist=True,prog_bar=False , on_step=False, on_epoch=True,logger=True)
            return {"Fscore_harmonious" : Fscore_harmonious}

    def on_validation_epoch_end(self):
            # Compute the average validation fscore for the epoch
            avg_val_fscore =torch.tensor(self.highest_Fscores).mean()

            # Check if this is the best validation fscore so far
            if avg_val_fscore > self.highest_Fscore:
                self.highest_Fscore = avg_val_fscore 
                self.log('best_val_fscore', self.highest_Fscore, prog_bar=True, logger=True)



    def test_step(self, batch, batch_idx):
        img,tar = batch
        if self.cfg.memory_module:
            pred,  featr, memr, featf, memf, attn_map_r, attn_map_f= self.ourmodel(img)
            loss, mem_lossr, mem_lossf, auc, f1_macro,  avg_p_mIoU, m_f1,iou_harmonious, Fscore_harmonious = self.do_prediction(pred,tar,img,featr, memr, featf, memf, attn_map_r, attn_map_f)
            self.log("Fscore_harmonious",Fscore_harmonious, on_step=True, on_epoch=True, prog_bar=True, logger=True) 
            return  {"test_Fscore":  Fscore_harmonious}

        else:
            pred= self.ourmodel(img)
            loss,auc, f1_macro, avg_p_mIoU, m_f1,iou_harmonious, Fscore_harmonious = self.do_prediction(pred,tar,img)
            self.log("Fscore_harmonious",Fscore_harmonious, on_step=False, on_epoch=True, prog_bar=True, logger=True) 
            return {"test_Fscore":  Fscore_harmonious}

    
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
    
    def do_prediction(self,pred,tar,img,  featr=None, memr=None, featf=None, memf=None, attn_map_r=None, attn_map_f=None):

        if pred.size(2) != img.size(2):
            pred = F.interpolate(pred, img.shape[2:], mode= 'bilinear', align_corners = True)
        y_score = F.softmax(pred, dim=1)[:,1,:,:]       
        #loss_iou = self.iou_loss(torch.unsqueeze(y_score, 1),torch.unsqueeze(tar.long().detach(), 1))
        auc = self.this_that_max(tar,y_score,matric='auc')
        f1_macro = self.this_that_max(tar,y_score,matric='f1', average= 'macro')
        size = pred.size()
        avg_p_mIoU, m_f1 = self.miou_f1(tar,pred,size)
        iou_harmonious  = compute_IoU(torch.unsqueeze(y_score, 1),torch.unsqueeze(tar, 1)) 
        Fscore_harmonious1 = FScore(torch.unsqueeze(y_score, 1),torch.unsqueeze(tar, 1)).cpu().detach()
        Fscore_harmonious2 = FScore(torch.unsqueeze(1-y_score, 1),torch.unsqueeze(tar, 1)).cpu().detach()
        Fscore_harmonious = max(Fscore_harmonious1, Fscore_harmonious2)
        loss_ce = self.criterion(pred, tar.long().detach()) 

        if self.cfg.memory_module:
            mem_lossr = self.sum_gather_spread(featr,memr,self.TrepletLoss_real)
            mem_lossf = self.sum_gather_spread(featf, memf, self.TrepletLoss_fake)
            total_loss =  self.cfg.main_loss_weight*loss_ce+ self.cfg.real_mem_loss_weight*mem_lossr+ self.cfg.fake_mem_loss_weight*mem_lossf #+ self.loss_wt_iou*loss_iou

            return [total_loss,loss_ce], mem_lossr, mem_lossf, auc, f1_macro, avg_p_mIoU, m_f1 , iou_harmonious, Fscore_harmonious #,loss_iou
        else:
            return loss_ce, auc, f1_macro, avg_p_mIoU, m_f1 , iou_harmonious, Fscore_harmonious #,loss_iou

    
    def miou_f1(self,label,pred,size):

        current_confusion_matrix = get_confusion_matrix(
                label,
                pred,
                size,
                 num_class=2, ignore=-1)

        pos = current_confusion_matrix.sum(1)  # ground truth label count
        res = current_confusion_matrix.sum(0)  # prediction count
        tp = np.diag(current_confusion_matrix)  # Intersection part
        IoU_array = (tp / np.maximum(1.0, pos + res - tp))  # Union part
        mean_IoU = IoU_array.mean()
        #avg_mIoU.update(mean_IoU)
        TN = current_confusion_matrix[0, 0]
        FN = current_confusion_matrix[1, 0]
        FP = current_confusion_matrix[0, 1]
        TP = current_confusion_matrix[1, 1]
        p_mIoU = 0.5 * (FN / np.maximum(1.0, FN + TP + TN)) + 0.5 * (FP / np.maximum(1.0, FP + TP + TN))
        avg_p_mIoU = np.maximum(mean_IoU, p_mIoU)

        precision = TP / np.maximum(1.0, TP + FP)
        recall = TP / np.maximum(1.0, TP + FN)
        m_f1 = 2 * (precision * recall) / np.maximum(1.0, precision + recall)


        return avg_p_mIoU, m_f1
    

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
