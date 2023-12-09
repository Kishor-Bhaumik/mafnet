import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import timm,pdb,math
from torch.nn.parameter import Parameter
from .aspp import build_aspp
from .srm import setup_srm_layer
import torch.nn.init as init
from .net import theta1,theta2,CustomNet,Decoder,Deconv,theta, SimpleEncoder, gumbel_softmax

#from .memory import Memory




class MemNet(nn.Module):
    def __init__(self, cfg, inplanes):
        super(MemNet, self).__init__()
        self.cfg = cfg
        # self.encoder = timm.create_model(self.cfg.encoder, pretrained= True, features_only=True, out_indices=[4])
        # self.encoder_srm = timm.create_model(self.cfg.encoder, pretrained= False, features_only=True, out_indices=[4])

        self.cfg = cfg
        self.encoder = SimpleEncoder()
        self.conv_srm = setup_srm_layer()
        self.encoder_srm = SimpleEncoder()

        self.real_mem = Parameter(torch.randn(self.cfg.mem_dim,self.cfg.fea_dim), requires_grad=True)  # M x C
        self.fake_mem = Parameter(torch.randn(self.cfg.mem_dim,self.cfg.fea_dim), requires_grad=True)  # M x C
        self.theta1 = theta()
        self.theta2 = theta()
        #self.decoder = Decoder()
        self.epsilon =1e-7 

        
        self.p_len = cfg.patch_size

        if self.cfg.with_srm :
            self.aspp = build_aspp(inplanes = inplanes*3, outplanes = self.cfg.aspp_outplane)
        else:
            self.aspp = build_aspp(inplanes = inplanes, outplanes = self.cfg.aspp_outplane )
        
            
        self.decoder =  nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU() )
        
        self.reset_mem_parameters()
        self._init_weight()
        

    def reset_mem_parameters(self):
        stdv = 1. / math.sqrt(self.real_mem.size(1))
        self.real_mem.data.uniform_(-stdv, stdv)
        stdv2 = 1. / math.sqrt(self.fake_mem.size(1))
        self.fake_mem.data.uniform_(-stdv2, stdv2)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inp, tar= None):

        x = self.encoder(inp) #torch.Size([16, 2048, 7, 7])

        if self.cfg.with_srm :
            x_srm = self.conv_srm(inp)
            x_srm = self.encoder_srm(x_srm)##torch.Size([16, 2048, 7, 7])

            updated_queryr, featr, memr = self.mem_attention(x_srm , tar , 'real', self.real_mem, self.training)
            updated_queryf, featf, memf = self.mem_attention(x_srm , tar , 'fake', self.fake_mem, self.training)

            x = torch.concat([x, updated_queryr, updated_queryf ], dim=1) #torch.Size([4, 192, 56, 56])
        
        x = self.aspp(x) #torch.Size([4, 512, 7, 7])

        out =self.decoder(x) #([16, 2, 28, 28])
        
        return out, featr, memr, featf, memf

        #mixed_up = self.LM3(updated_queryr,updated_queryf) #([16, 128, 56, 56])
        concate = torch.cat((updated_queryr,updated_queryf),1) #torch.Size([16, 1024, 56, 56])
        out =self.decoder(concate) #([16, 2, 28, 28])

        
        return out, featr, memr, featf, memf
    

    def LM3 (self, feat1,feat2):

        feat1 =self.theta1(feat1)
        feat2 = self.theta2(feat2)
        
        return torch.add(feat1,feat2)


    def mem_attention(self, feat , target_mask,Type , mem, train):
        
        batch_size, dims,h,w = feat.size() # b X d X h X w

        if train:
            #assert target_mask != None
            target_mask = target_mask.detach()
            target_mask= F.avg_pool2d(target_mask, kernel_size = self.p_len, stride=self.p_len)
            target_mask = (target_mask>0.5).int().float()
            
            if Type =='real':
                target_mask =torch.logical_not(target_mask).int() # ) changes 0 to 1 and 1 changes to 0
                
            ## 0's can be replaced with very small integer
            target_mask= torch.where(target_mask == 0, torch.tensor(self.epsilon, dtype=target_mask.dtype).to(target_mask.device), target_mask)
            feat = feat* target_mask.unsqueeze(1)
            query = F.normalize(feat, dim=1)
            updated_memory = self.update(query ,mem,train)
            mem = updated_memory
            feat = query
        
        att2 = self.read(feat, mem)
        #feat = feat.permute(0,3,1,2)
        attn_sim = self.cosin_sim(feat, att2)
        updated_query = feat* attn_sim.unsqueeze(1)
        # mem_loss = gathering_loss+ spreading_loss
        return updated_query , feat, mem


    def cosin_sim (self, old_feat, new_feat):
        old_feat =self.theta1(old_feat)
        new_feat = self.theta2(new_feat)
        similarity = F.cosine_similarity(old_feat,new_feat, dim=1)
        similarity_squeezed = torch.squeeze(similarity, dim=1)
    
        return similarity_squeezed

    def reshape(self,tensor,train):

        if not train:
            tensor = F.normalize(tensor, dim=1)
            return tensor #.permute(0,2,3,1)  # b X h X w X d
        else: return tensor



    def update(self, query, keys,train):
        query = F.normalize(query, dim =1).permute(0,2,3,1)
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
        score_memory =F.softmax(score,dim=1)
        
        # score_query = F.gumbel_softmax(score, tau=1, hard=False, dim =0) 
        # score_memory = F.gumbel_softmax(score, tau=1, hard=False, dim =1) 

        return score_query , score_memory

    def get_update_query(self, mem, max_indices, update_indices, score, query, train):
        

        m, d = mem.size()
        if train:
            query_update = torch.zeros((m,d)).to(query.device)
            for i in range(m):
                idx = torch.nonzero(max_indices.squeeze(1)==i)
                a, _ = idx.size()
                #ex = update_indices[0][i]
                if a != 0:
                    query_update[i] = torch.sum(((score[idx,i] / torch.max(score[:,i])) *query[idx].squeeze(1)), dim=0)
                else:
                    query_update[i] = 0 

            
            return query_update 
    
    def read(self, query, updated_memory):
        query = query.permute(0,2,3,1)   
        batch_size, h,w,dims = query.size() # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_score(updated_memory, query)

        att2= torch.matmul(softmax_score_memory.detach(), updated_memory) # (b X h X w) X d 
        att2 =att2.view(batch_size, h, w,dims)
        att2 = att2.permute(0,3,1,2)
        
        return att2

