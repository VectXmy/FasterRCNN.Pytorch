'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-11-25
@Email: xxxmy@foxmail.com
'''

import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F


def gen_base_anchors(base_size=16,ratios=(0.5,1.,2.),scales=(8,16,32)):
    '''
    return
    base_anchors shape[s*r,4] format (y1,x1,y2,x2)
    '''
    base_anchors=torch.zeros((len(ratios)*len(scales),4),dtype=torch.float32)
    cx=base_size/2.
    cy=base_size/2.
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h=base_size*scales[j]*math.sqrt(ratios[i])
            w=base_size*scales[j]*math.sqrt(1./ratios[i])
            index=i*len(scales)+j
            base_anchors[index, 0] = cy - h / 2.
            base_anchors[index, 1] = cx - w / 2.
            base_anchors[index, 2] = cy + h / 2.
            base_anchors[index, 3] = cx + w / 2.
    return base_anchors

def shift_anchors(base_anchors,feature_stride,fmap_size):
    h,w=fmap_size
    shifts_y=torch.arange(0,h*feature_stride,feature_stride,dtype=base_anchors.dtype)
    shifts_x=torch.arange(0,w*feature_stride,feature_stride,dtype=base_anchors.dtype)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    shifts=torch.stack([shift_y,shift_x,shift_y,shift_x],dim=-1)#(h*w,4)
    A=base_anchors.shape[0]#(s*r,4)
    shifts_num=shifts.shape[0]#(h*w,4)
    anchors=base_anchors.reshape((1,A,4))+\
            shifts.reshape((1,shifts_num,4)).permute(1,0,2)#(1,A,4)+(h*w,1,4)-->(h*w,A,4)
    anchors=anchors.reshape((shifts_num*A,4)).float()#(h*w*A,4)
    return anchors

def anchor2box(anchors,pred_locs):
    '''
    Args
    pred_locs [h*w*A,4] offsets
    anchors [h*w*A,4]
    Returns
    boxes [h*w*A,4]
    '''
    if anchors.shape[0]==0:
        return torch.zeros((0,4),dtype=pred_locs.dtype)
    
    h_a=anchors[...,2]-anchors[...,0]#(h*w*A,)
    w_a=anchors[...,3]-anchors[...,1]
    cy_a=anchors[...,0]+0.5*h_a
    cx_a=anchors[...,1]+0.5*w_a
    
    t_y=pred_locs[...,0]#(h*w*A,)
    t_x=pred_locs[...,1]
    t_h=pred_locs[...,2]
    t_w=pred_locs[...,3]

    cy_b=t_y*h_a+cy_a
    cx_b=t_x*w_a+cx_a
    h_b=torch.exp(t_h)*h_a
    w_b=torch.exp(t_w)*w_a

    #(cy,cx,h,w)-->(y1,x1,y2,x2)
    y1_b=cy_b-0.5*h_b
    x1_b=cx_b-0.5*w_b
    y2_b=cy_b+0.5*h_b
    x2_b=cx_b+0.5*w_b#(h*w*A,)

    boxes=torch.stack([y1_b,x1_b,y2_b,x2_b],dim=-1)#(h*w*A,4)

    return boxes

def boxes_nms(boxes,scores,thr):
    '''
    boxes: [?,4]
    scores: [?]
    '''
    if boxes.shape[0]==0:
        return torch.zeros(0,device=boxes.device).long()
    assert boxes.shape[-1]==4
    y1,x1,y2,x2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1+1)*(y2-y1+1)
    order=scores.sort(0,descending=True)[1]
    keep=[]
    while order.numel()>0:
        if order.numel()==1:
            i=order.item()
            keep.append(i)
            break
        else:
            i=order[0].item()
            keep.append(i)
        
        xmin=x1[order[1:]].clamp(min=float(x1[i]))
        ymin=y1[order[1:]].clamp(min=float(y1[i]))
        xmax=x2[order[1:]].clamp(max=float(x2[i]))
        ymax=y2[order[1:]].clamp(max=float(y2[i]))
        inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        idx=(iou<=thr).nonzero().squeeze()
        if idx.numel()==0:
            break
        order=order[idx+1]
        return torch.LongTensor(keep)

class RegionProposalNetwork(nn.Module):
    def __init__(self,in_channels=512,mid_channels=512,
                anchor_ratios=(0.5,1,2),
                anchor_scales=(8,16,32),feature_stride=16):
        super().__init__()

        self.base_anchors=gen_base_anchors()
        self.anchor_num=self.base_anchors.shape[0]
        self.conv1=nn.Conv2d(in_channels,mid_channels,3,1,1)
        self.rpn_scores=nn.Conv2d(mid_channels,self.anchor_num*2,1,1)
        self.rpn_locs=nn.Conv2d(mid_channels,self.anchor_num*4,1,1)
        self.feature_stride=feature_stride
        self.apply(self.init_conv_uniform)

        self.proposal_creator=ProposalCreator(self)

    def init_conv_uniform(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.uniform_(module.weight,0.,0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
    
    def forward(self,x,img_size,scale=1.0):
        '''
        x fmaps
        img_size (2,) list
        scale float
        '''
        fmap_size=x.shape[2:]
        batch_size=x.shape[0]

        anchors=shift_anchors(self.base_anchors,self.feature_stride,fmap_size)#(h*w*A,4)
        anchors.to(x.device)

        conv_out=F.relu(self.conv1(x),True)#(b,512,h,w)
        rpn_pred_scores=self.rpn_scores(conv_out)#(b,A*2,h,w)
        rpn_pred_locs=self.rpn_locs(conv_out)#(b,A*4,h,w)
        rpn_pred_locs=rpn_pred_locs.permute(0,2,3,1).reshape((batch_size,-1,4))#(b,h*w*A,4)
        rpn_pred_scores=rpn_pred_scores.permute(0,2,3,1).reshape((batch_size,-1,2))
        rpn_pred_scores_softmax=F.softmax(rpn_pred_scores)#(b,h*w*A,2)
        rpn_pred_fg_scores=rpn_pred_scores_softmax[...,1]#(b,h*w*A,)

        rois=[]
        rois_inds=[]#for example [0,0,1,1,1,2,2,3,3,...] different number for each batch dim
        for batch_index in range(batch_size):
            rois_b=self.proposal_creator(rpn_pred_locs[batch_index],
                                        rpn_pred_fg_scores[batch_index],
                                        anchors,
                                        img_size,
                                        scale=scale)#(?,4)
            index_b=batch_index*torch.ones(len(rois_b),device=rois_b.device).long()
            rois.append(rois_b)
            rois_inds.append(index_b)
        rois=torch.cat(rois,dim=0)#(??,4)
        rois_inds=torch.cat(rois_inds)#(??,)
        
        return rois,rois_inds,anchors,rpn_pred_scores,rpn_pred_locs


class ProposalCreator():
    def __init__(self,
                parent_model,
                nms_thr=0.7,
                train_nms_num=(12000,2000),
                test_nms_num=(6000,300),
                min_size=16
                ):
        self.parent_model=parent_model
        self.nms_thr=nms_thr
        self.train_nms_num=train_nms_num
        self.test_nms_num=test_nms_num
        self.min_size=min_size
    
    def __call__(self,pred_locs,pred_fg_scores,anchors,img_size,scale=1.0):
        '''
        Args
        pred_locs [h*w*A,4]
        pred_fg_scores [h*w*A,]
        anchors [h*w*A,4]
        img_size [2,] list
        scale float
        '''
        if self.parent_model.training:
            nms_num=self.train_nms_num
        else:
            nms_num=self.test_nms_num

        rois=anchor2box(anchors,pred_locs)#(h*w*A,4)
        #clip boxes
        rois[...,[0,2]]=rois[...,[0,2]].clamp(min=0,max=img_size[0])
        rois[...,[1,3]]=rois[...,[1,3]].clamp(min=0,max=img_size[1])
        #remove boxes that h or w less than min_size
        min_size=self.min_size*scale
        rois_h=rois[...,2]-rois[...,0]#(h*w*A,)
        rois_w=rois[...,3]-rois[...,1]
        keep_mask=(rois_h>=min_size)&(rois_w>=min_size)#(h*w*A,)
        rois=rois[keep_mask]#(?,4)
        assert rois.shape[-1]==4 and len(rois.shape)==2

        scores=pred_fg_scores[keep_mask]#(?,)
        #nms rois
        nms_before_inds=scores.argsort(0,descending=True)[:nms_num[0]]
        print("nms_before",len(nms_before_inds))
        rois=rois[nms_before_inds]
        nms_keep_inds=boxes_nms(rois,scores,self.nms_thr)[:nms_num[1]]
        rois=rois[nms_keep_inds]

        return rois

def __test__():
    base_anchors=gen_base_anchors()
    anchors=shift_anchors(base_anchors,32,(16,16))
    print("base_anchors",base_anchors.shape," anchors",anchors.shape)
    pred_locs=torch.ones_like(anchors)
    pred_fg_scores=torch.ones([anchors.shape[0],],dtype=torch.float32)
    print("pred_locs",pred_locs.shape," pred_fg_scores",pred_fg_scores.shape)
    model=RegionProposalNetwork().train()
    a=ProposalCreator(model)
    rois=a(pred_locs,pred_fg_scores,anchors,(512,512))
    print("rois",rois.shape,rois)

if __name__ == "__main__":
    __test__()






        