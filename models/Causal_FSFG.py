import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4,ResNet
from .backbones.FSRM import FSRM
from .backbones.IMFR import IMFR

from .causal_modules import IMSE_Residual


class CausalFSFG(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False):
        
        super().__init__()
        

        self.resolution = 5*5
        if resnet:
            self.num_channel = 640
            self.feature_extractor = ResNet.resnet12()

            backbone_out_channels = [64, 160, 320, 640]
            embed_dim = 256
            self.imse = IMSE_Residual(in_channels_list=backbone_out_channels, reduce_dim=128)
            self.final_feat_channels = 640 
            
            self.num_channel = embed_dim
            
        else:
            self.num_channel = 64
            self.feature_extractor = Conv_4.BackBone(self.num_channel)            

            backbone_out_channels = [64, 64, 64, 64]
            embed_dim = 128
            self.imse = IMSE_Residual(in_channels_list=backbone_out_channels, reduce_dim=128)
            self.final_feat_channels = 64
            
            self.num_channel = embed_dim

        self.fsrm = FSRM(
                sequence_length=self.resolution,
                embedding_dim=self.final_feat_channels,
                num_layers=1,
                num_heads=1,
                mlp_dropout_rate=0.,
                attention_dropout=0.,
                positional_embedding='sine')

        self.imfr = IMFR(
            hidden_size=self.final_feat_channels, 
            inner_size=self.num_channel, 
            num_patch=self.resolution, 
            drop_prob=0.1)
        
        self.dim = self.final_feat_channels * 5 * 5

        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]),requires_grad=True)
            

    def get_feature_vector(self,inp):

        batch_size = inp.size(0)
        f1, f2, f3, f4 = self.feature_extractor(inp)
        feature_map = self.imse([f1, f2, f3, f4])

        feature_map = self.fsrm(feature_map).transpose(1, 2).view(batch_size, 
                                                                #   self.num_channel,
                                                                self.final_feat_channels, 
                                                                  5, 5)

        return feature_map
    

    def get_neg_l2_dist(self,inp,way,shot,query_shot):

        feature_map = self.get_feature_vector(inp) 
        support = feature_map[:way*shot].view(way, shot, *feature_map.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
        query = feature_map[way*shot:]
        
        sq_similarity, qs_similarity = self.imfr(support, query)

        l2_dist = self.w1*sq_similarity + self.w2*qs_similarity
        
        return l2_dist



    
    def meta_test(self,inp,way,shot,query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                        way=way,
                                        shot=shot,
                                        query_shot=query_shot)

        _,max_index = torch.max(neg_l2_dist,1)

        return max_index


    def forward(self,inp):

        logits = self.get_neg_l2_dist(inp=inp,
                                        way=self.way,
                                        shot=self.shots[0],
                                        query_shot=self.shots[1])
        logits = logits/self.dim*self.scale

        log_prediction = F.log_softmax(logits,dim=1)

        return log_prediction