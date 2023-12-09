import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import timm,pdb
from torch.nn.parameter import Parameter
from .aspp import build_aspp
from .srm import setup_srm_layer
import torch.nn.init as init


class theta1(nn.Module):
    def __init__(self):
        super(theta1, self).__init__()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x
    
class theta2(nn.Module):
    def __init__(self):
        super(theta2, self).__init__()

        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x
#from .memory import Memory


class MemNet(nn.Module):
    def __init__(self, cfg, inplanes):
        super(MemNet, self).__init__()
        self.cfg = cfg
        self.encoder = timm.create_model(self.cfg['model_params']['encoder'], pretrained= True, features_only=True, out_indices=[4])
        self.conv_srm = setup_srm_layer()
        self.encoder_srm = timm.create_model(self.cfg['model_params']['encoder'], pretrained= False, features_only=True, out_indices=[4])
        self.real_mem = Parameter(torch.Tensor(self.cfg['model_params']['mem_dim'], self.cfg['model_params']['fea_dim']))  # M x C
        self.fake_mem = Parameter(torch.Tensor(self.cfg['model_params']['mem_dim'], self.cfg['model_params']['fea_dim']))  # M x C
        self.theta1 = theta1()
        self.theta2 = theta2()
        self.epsilon =1e-7 
        # self.memory = Memory()
        self.TrepletLoss = torch.nn.TripletMarginLoss(margin=self.cfg['model_params']['triplet_margin'])
        self.loss_mse = torch.nn.MSELoss()

        self.deconv = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=8, stride=8, padding=0)

        init.xavier_uniform_(self.real_mem)
        init.xavier_uniform_(self.fake_mem)

        
        self.p_len = cfg['dataset_params']['patch_size']

        if self.cfg['global_params']['with_srm'] == True:
            self.aspp = build_aspp(inplanes = inplanes*2, outplanes = self.cfg['model_params']['aspp_outplane'] )
        else:
            self.aspp = build_aspp(inplanes = inplanes, outplanes = self.cfg['model_params']['aspp_outplane'] )
        
            
        self.decoder = nn.Sequential(
                                        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2, stride=2)
                                    )
        self.projection = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=1),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 256, kernel_size=1)
            )
    def forward(self, inp, tar= None):
        x = self.encoder(inp)[0]

        if self.cfg['global_params']['with_srm'] == True:
            x_srm = self.conv_srm(inp)
            x_srm = self.encoder_srm(x_srm)[0]
            x = torch.concat([x, x_srm], dim=1) #torch.Size([4, 4096, 7, 7])
        
        x = self.aspp(x) #torch.Size([4, 512, 7, 7])

        feat = self.deconv(x)
        
            
        updated_queryr, mem_lossr = self.mem_attention(feat , tar , 'real', self.real_mem, self.training)
        updated_queryf, mem_lossf = self.mem_attention(feat , tar , 'fake', self.fake_mem, self.training)

        mixed_up = self.LM3(updated_queryr,updated_queryf)
        #pdb.set_trace()

        out =self.decoder(mixed_up)

        # fake_sim = self.cosin_sim(feat, updated_queryf)
        # real_sim = self.cosin_sim(feat, updated_queryr)

            
        return out, mem_lossr ,mem_lossf
    

    def LM3 (self, feat1,feat2):

        feat1 =self.theta1(feat1)
        feat2 = self.theta2(feat2)
        
        return torch.add(feat1,feat2)

    def cosin_sim (self, old_feat, new_feat):
        b,c,h,w = old_feat.size()
        new_feat = new_feat.view(b,c,h,w)
        old_feat =self.theta1(old_feat)
        new_feat = self.theta2(new_feat)
        similarity = F.cosine_similarity(old_feat,new_feat, dim=1)
        similarity_squeezed = torch.squeeze(similarity, dim=1)

        return similarity_squeezed




    def mem_attention(self, feat , target_mask,Type , mem, train):
        
        batch_size, dims,h,w = feat.size() # b X d X h X w

        if train:

            assert target_mask != None
            target_mask = target_mask.detach()
            target_mask= F.avg_pool2d(target_mask, kernel_size = self.p_len, stride=self.p_len)
            target_mask = (target_mask>0.5).int().float()
            
            if Type =='real':
                target_mask =torch.logical_not(target_mask).int() # ) changes 0 to 1 and 1 changes to 0
                
            ## or 0's can be replaced with very small integer
            target_mask= torch.where(target_mask == 0, torch.tensor(self.epsilon, dtype=target_mask.dtype).to(target_mask.device), target_mask)
            feat = feat* target_mask.unsqueeze(1)
            query = F.normalize(feat, dim=1)
            query = query.permute(0,2,3,1) # b X h X w X d
            updated_memory = self.update(query ,mem,train)
            mem = updated_memory
            feat = query
        
        feat = self.reshape(feat,train)

        gathering_loss = self.gather_loss(feat,mem)
            #spreading_loss
        spreading_loss = self.spread_loss(feat, mem)

        updated_query = self.read(feat, mem)

        mem_loss = gathering_loss+ spreading_loss

        #updated_query, updated_memory, softmax_score_query, softmax_score_memory, gathering_loss, spreading_loss = self.memory(feat,mem,train)
        
        return updated_query, mem_loss

    def reshape(self,tensor,train):

        if not train:
            tensor = F.normalize(tensor, dim=1)
            return tensor.permute(0,2,3,1)  # b X h X w X d
        else: return tensor



    def update(self, query, keys,train):

        batch_size, h,w,dims = query.size() # b X h X w X d 

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)
        
        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        
        query_update = self.get_update_query(keys, gathering_indices, updating_indices, softmax_score_query, query_reshape,train)
        updated_memory = F.normalize(query_update + keys, dim=1)

      
        return updated_memory #.detach()

    def get_score(self, mem, query):
        bs, h,w,d = query.size()
        
        #pdb.set_trace()
        m, d = mem.size()
        
        score = torch.matmul(query, torch.t(mem))# b X h X w X m
        score = score.view(bs*h*w, m)# (b X h X w) X m
        
        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score,dim=1)

        return score_query , score_memory



    def spread_loss(self,query, keys):
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        #1st, 2nd closest memories
        pos = keys[gathering_indices[:,0]]
        neg = keys[gathering_indices[:,1]]

        spreading_loss = self.TrepletLoss(query_reshape,pos.detach(), neg.detach())

        return spreading_loss
        
    def gather_loss(self, query, keys):
        
        
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = self.loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss
    

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        
        # pdb.set_trace()
        m, d = mem.size()
        if train:
            query_update = torch.zeros((m,d)).cuda()
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                #ex = update_indices[0][i]
                if a != 0:
                    #random_idx = torch.randperm(a)[0]
                    #idx = idx[idx != ex]
#                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                    #random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
                else:
                    query_update[i] = 0 
                    #random_update[i] = 0
            
            return query_update 
        

    def read(self, query, updated_memory):
        
        batch_size, h,w,dims = query.size() # b X h X w X d

        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        query_reshape = query.contiguous().view(batch_size*h*w, dims)
        
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d   ##equation 2 in the paper
        updated_query = torch.cat((query_reshape, concat_memory), dim = 1) # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2*dims)
        updated_query = updated_query.permute(0,3,1,2)
        
        return updated_query
    
