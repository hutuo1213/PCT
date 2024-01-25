"""
Interaction head and its submodules

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import torch
import torch.nn.functional as F
import random

from torch import nn, Tensor
from typing import List, Optional, Tuple
from collections import OrderedDict
import detr
from detr.models.transformer import TransformerDecoderLayer,TransformerEncoderLayer
import pocket
from torchvision.ops import roi_align
import numpy as np
# from torchvision.transforms.functional import hflip
# from pocket.models.transformers import CrossAttentionLayer

class InteractionHead(nn.Module):
    """
    Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_pair_predictor: nn.Module
        Module that classifies box pairs
    hidden_state_size: int
        Size of the object features
    representation_size: int
        Size of the human-object pair features
    num_channels: int
        Number of channels in the global image features
    num_classes: int
        Number of target classes
    human_idx: int
        The index of human/person class
    object_class_to_target_class: List[list]
        The set of valid action classes for each object type
        ##$$object_to_target:80list:每个物体对应的有效动词
    """
    def __init__(self,
        box_pair_predictor: nn.Module,
        hidden_state_size: int, representation_size: int,
        num_channels: int, num_classes: int, human_idx: int,
        object_class_to_target_class: List[list]
    ) -> None:
        super().__init__()

        self.box_pair_predictor = box_pair_predictor##linear(512*2,117)
        # self.cls_head = nn.Linear(512,117)
        # self.box_pair_predictor = nn.Linear(256,117)  ##linear(512*2,117)
        self.hidden_state_size = hidden_state_size#256
        self.representation_size = representation_size#512
        self.num_classes = num_classes#117
        self.human_idx = human_idx##0
        self.object_class_to_target_class = object_class_to_target_class##每个对象的有效谓词

        self.linear2048 = nn.Sequential(nn.Linear(2048,256),nn.LayerNorm(256),nn.ReLU())
        self.linear2048_h = nn.Sequential(nn.Linear(2048, 256), nn.LayerNorm(256), nn.ReLU())

        self.decode = TransformerDecoderLayer(512, 8, activation="gelu")
        # self.decode2 = TransformerDecoderLayer(512, 8, activation="gelu")
        self.comp_layer = pocket.models.TransformerEncoderLayer(
            hidden_size=512,
            return_weights=True)

        self.memory = nn.Sequential(nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU())

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:
        #x,y为人的索引和物的索引
        ##$$:prior_h,prior_o:tensor(15,117),列：交互对，横：动词；x有3个人，y有6个物
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)#[n_h,117]
        prior_o = torch.zeros_like(prior_h)#[n_h,117]

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        # p = 1.0 if self.training else 0.0
        # print(f'p == {p}')
        # p = 1.0 if self.training else 1.0
        ##$$s_h、s_o取出每对交互人的得分、物的得分
        s_h = scores[x].pow(p)
        # s_h = scores[x]
        s_o = scores[y].pow(p)
        # s_o = scores[y]

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        # 将对象类索引映射到目标类索引
        # 对象类索引到目标类索引是一对多映射
        ##$$15对交互中，物体对应的有效动词list
        target_cls_idx = [self.object_class_to_target_class[obj.item()]#物的类别分别对应的有效动作
            for obj in object_class[y]]#object_class[y]：符合条件的物的类别
        #target_cls_idx#0[4,5,6];1[7,8,9,10];2[11,12]
        # Duplicate box pair indices for each target class每个目标类的重复框对索引
        # 80个list，每个list对应该物体可能的动词----#80个list，每个list长度为该物体的有效动词，每个list用list的idx填充
        #0[0,0,0];1[1,1,1,1];2[2,2]--[0,0,0,1,1,1,1,2,2]物体对应的索引
        ##$$15对交互对中，第i个物体对应对应有效动词list全置为i,pair_idx:tensor(114),(出现重复)
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]#每个0,1,2...对象可能对应的动作数量#x,y的维度相同一一对应
        # Flatten mapped target indices#[4,5,6,7,8,9,10,11,12]动词索引
        ##$$15对交互对中，每个物体对应的有效动词list展平,与pair_idx对齐flat_target_idx:tensor(114)
        flat_target_idx = [t for tar in target_cls_idx for t in tar]#发生的动作
        ##$$pair_idx表示第i个交互，flat_target_idx表示第i个交互对应的动词索引
        ##$$prior_h、prior_o：tensor(15,117) = 第i个交互人的得分；第i个交互物的得分（第i个交互对应的可能动词全被置为第i个交互的人、物得分）
        ##$$s_h、s_o取出每对交互人的得分、物的得分
        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]#人的得分，复制动作（与物发生有效交互）数量次

        prior_h = prior_h.masked_fill(prior_h != 0, 1.0)
        # print(prior_h)
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]#物的得分，复制动作（与物发生有效交互）数量次
        prior_o = prior_o.masked_fill(prior_o != 0, 1.0)
        #prior_h[pair_idx, flat_target_idx]：[n_h,117]
        return torch.stack([prior_h, prior_o])#[2,15,117]

    def forward(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict],memory:Tensor,m_mask:Tensor):
    # def forward(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]):
        # features[-1].tensors：N,2048,H,W；#image_sizes = [B,2]batch中真实图片的大小
        # region_props：返回符合要求的bbox、sc、lb、hs（特征向量）#得分大于0.2的索引(keep=[B*Q])，前面是人，后面是物
        #keep = torch.cat([keep_h, keep_o])  # 得分大于0.2的索引，前面是人，后面是物
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        image_shapes: Tensor
            (B, 2) Image shapes, heights followed by widths
        region_props: List[dict]
            Region proposals with the following keys
            `boxes`: Tensor
                (N, 4) Bounding boxes
            `scores`: Tensor
                (N,) Object confidence scores
            `labels`: Tensor
                (N,) Object class indices
            `hidden_states`: Tensor
                (N, 256) Object features
        """
        device = features.device
        # global_features = self.avg_pool(features).flatten(start_dim=1)
        # memory,[b,c,h,w]--[hw,b,c]

        memory_k = memory.flatten(2).permute(2,0,1)#decoder格式
        m_mask = m_mask.flatten(1)
        memory_c = self.memory(memory_k)

        # memory_c2 = self.memory2(memory_k2)

        #### src----------B,2048,H,W,
        ##B,2048,1,1----B,2048
        # cls_collated = []
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        pairwise_tokens_collated = []
        attn_maps_collated = []

        for b_idx, props in enumerate(region_props):#拆分batch
            # print(b_idx)
            #$$region_props;   keep与查询是对齐的
            #$$8list:{boxes=bx[keep],scores=sc[keep],labels=lb[keep],hidden_states=hs[keep]}，前面是人，后面是物
            #$$keep:tensor([0, 1, 25, 2, 3, 4], device='cuda:0')第一张图片（list：0）收集6个信息
            boxes = props['boxes']#[6,4]
            # boxes_copy = boxes.clone()
            scores = props['scores']#[6]
            labels = props['labels']##$$labels:tensor([ 0,  0,  0, 43, 71, 43], device='cuda:0')
            unary_tokens = props['hidden_states']#tensor(6,256)
            # #得分大于0.2的索引，前面是人，后面是物# sc >= 0.2##keep=[Q]
            is_human = labels == self.human_idx#bool,与lb维度相同，人的索引被置为ture
            n_h = torch.sum(is_human); n = len(boxes)#n_h=人的数量，n=人和物一共的数量
            ##$$n_h = 3; n = 6
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):#验证前n_h个标签是否为人
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; unary_tokens = unary_tokens[perm]

            # Skip image when there are no valid human-object pairs
            # 在没有有效的人 - 对象对时跳过图像
            if n_h == 0 or n <= 1:
                pairwise_tokens_collated.append(torch.zeros(#tensor([], size=(0, 1024))
                    0,512,
                    # 0,256,
                    device=device)
                )
                ##$$
                # cls_collated.append(torch.zeros(1,512, device=device))
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))#tensor([])
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))#tensor([])
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))#tensor([])
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                #self.num_classes=117#tensor([], size=(0, 2, 117))
                continue#再次进入循环，不进行后面的处理
            ############################################
            # Get the pairwise indices
            #获取成对索引
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # x, y = torch.meshgrid(
            #     torch.arange(n_h, device=device),
            #     torch.arange(n, device=device)
            # )
            '''x, y = torch.meshgrid(                               
                torch.arange(3),
                torch.arange(3)
            )
                tensor([[0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2]])
                tensor([[0, 1, 2],
                        [0, 1, 2],
                        [0, 1, 2]])
            x = x.flatten(); y = y.flatten()
                tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
                tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
            torch.nonzero(torch.logical_and(x != y, x < 3))
                tensor([[0, 1],
                        [0, 2],
                        [1, 0],
                        [1, 2],
                        [2, 0],
                        [2, 1]])
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
                x < 2
                tensor([0, 0, 1, 1])
                tensor([1, 2, 0, 2])

                '''
            # Valid human-object pairs
            #计算输入张量的逻辑与bool# torch.nonzero返回不为0的索引int#unbind(1)沿维度1进行切片
            #x_keep代表图片中所有人符合要求的竖索引：y_keep代表图片中所有物(含其他人)符合要求的横索引；
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)

            boxeees = boxes / 32#32倍下采样映射到最后一层特征图上
            # boxeees = boxes / 64  # 64倍下采样映射到最后一层特征图上
            # union = [union_box(boxeees[x_keep],boxeees[y_keep])]
            # u_roi = roi_align(features[b_idx:b_idx + 1], union, 1)#[15,2048,1,1]
            # u_roit = torch.squeeze(u_roi)#[15,2048]
            # u_r = self.linear2048_u(u_roit)#[15,512]
            # boxeees = boxes / 32
            boxees = [boxeees]
            roio = roi_align(features[b_idx:b_idx + 1], boxees, 1).squeeze()# [6,2048,1,1]-[6,2048]
            roi_o = self.linear2048(roio[n_h:])
            roi_h = self.linear2048_h(roio[:n_h])
            roi = torch.cat([roi_h, roi_o], dim=0)

            if len(x_keep) == 0:
                # Should never happen, just to be safe
                #永远不应该发生，只是为了安全
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()

            a = torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1)
            b = torch.cat([roi[x_keep], roi[y_keep]],dim=1)
            ab = a+b
            # ab = torch.cat([unary_tokens[x_keep], unary_tokens[y_keep]], 1)

            # # # # # # # # # # # # # #Q[100,batch,256]K,V[H*W,batch,256]mask[batch,h*w]
            ab = ab.unsqueeze(dim=1)  # [15,1,512]
            ab = self.decode(ab, memory_c[:, b_idx:b_idx + 1], memory_key_padding_mask=m_mask[b_idx:b_idx + 1])
            # ab = self.decode2(ab, memory_c[:, b_idx:b_idx + 1], memory_key_padding_mask=m_mask[b_idx:b_idx + 1])
            ab = ab.squeeze(dim=1)
            pairwise_tokens, pairwise_attn = self.comp_layer(ab)
            # pairwise_tokens, pairwise_attn = self.comp_layer2(pairwise_tokens)

            pairwise_tokens_collated.append(pairwise_tokens)#拟[N,1024]
            boxes_h_collated.append(x_keep)#人的索引
            boxes_o_collated.append(y_keep)#物(含其他人)的索引
            object_class_collated.append(labels[y_keep])#物的类别(含其他人)
            # The prior score is the product of the object detection scores
            #先前的分数是对象检测分数的乘积
            prior_collated.append(self.compute_prior_scores(#[2,15,117]：交互三元组中人、物对应动词的分数
                x_keep, y_keep, scores, labels)#tensor：15，15，6，6
            )
            # attn_maps_collated.append((coop_attn,pairwise_attn))#拟注意力
            attn_maps_collated.append(())

        # box_collated = torch.cat(box_collated)
        # logits_s = self.spread(logits_s_collated)
        # logits_s = self.space(logits_s_collated)

        # cls_collated = torch.cat(cls_collated) # 整个batch[B*N,1024]
        # cls_coll = self.box_pair_predictor(cls_collated)
        pairwise_tokens_collated = torch.cat(pairwise_tokens_collated)#整个batch[B*N,1024]
        logits = self.box_pair_predictor(pairwise_tokens_collated)##linear(512*2,117)#[B*N,117]
        # logits_s = self.suppressor(pairwise_tokens_collated)
        # prebox = self.boxlabel(pairwise_tokens_collated)
        ##$$list0：bh、bo、obj：15对交互中人、物的边框索引、物的类别,人3个，物6个；list1:255list2:15list3:39
        return logits, prior_collated, \
            boxes_h_collated, boxes_o_collated, object_class_collated, attn_maps_collated#,cls_coll

    # def bbox_area(self, bbox):
    #     width = bbox[:, 2] - bbox[:, 0]
    #     height = bbox[:, 3] - bbox[:, 1]
    #     area = width * height
    #     return area
