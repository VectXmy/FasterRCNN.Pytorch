'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-12-05
@Email: xxxmy@foxmail.com
'''

import torch
import torch.nn as nn
import numpy as np
import random

def anchor2loc(anchors,gt_boxes):
    assert anchors.shape==gt_boxes.shape
    h_a=anchors[...,2]-anchors[...,0]
    w_a=anchors[...,3]-anchors[...,1]
    cy_a=anchors[...,0]+0.5*h_a
    cx_a=anchors[...,1]+0.5*w_a

    h_gt=gt_boxes[...,2]-gt_boxes[...,0]#(?,)
    w_gt=gt_boxes[...,3]-gt_boxes[...,1]
    cy_gt=gt_boxes[...,0]+0.5*h_gt
    cx_gt=gt_boxes[...,1]+0.5*w_gt

    t_y=(cy_gt-cy_a)/h_a
    t_x=(cx_gt-cx_a)/w_a
    t_h=torch.log(h_gt/h_a)
    t_w=torch.log(w_gt/w_a)

    locs=torch.stack([t_y,t_x,t_h,t_w],dim=-1)#(?,4)
    return locs

def boxes_ious(boxes1,boxes2):
    '''
    boxes1 [m,4] format (y1,x1,y2,x2)
    boxes2 [n,4]
    '''
    assert boxes1.shape[1]==4 and boxes2.shape[1]==4
    tl=torch.max(boxes1[:,None,:2],boxes2[None,:,:2])#(m,n,2)
    br=torch.min(boxes1[:,None,2:],boxes2[None,:,2:])
    hw=(br-tl).clamp(min=0)
    areas_inter=torch.prod(hw,dim=-1)#(m,n)
    area1=torch.prod(boxes1[:,2:]-boxes1[:,:2],dim=-1)#(m,)
    area2=torch.prod(boxes2[:,2:]-boxes2[:,:2],dim=-1)#(n,)
    ious=areas_inter/(area1[:,None]+area2[None,:]-areas_inter)#(m,n)

    return ious

class FasterRCNN(nn.Module):
    def __init__(self,mode,extractor,rpn,head,config=None):
        super().__init__()
        self.extractor=extractor
        self.rpn=rpn
        self.head=head
        self.mode=mode
        self.config=config
        self.proposal_target_creator=ProposalTargetCreator()
        self.anchor_target_creator=AnchorTargetCreator()
        self.compute_rpn_loss=RPNLOSS()
        self.compute_roi_loss=ROILOSS()

    def forward(self,x,scale=1.):
        if self.mode=="training":
            img_size=x["imgs"].shape[2:] # h,w
            gt_boxes=x["boxes"]#(b,m,4)
            gt_labels=x["labels"]#(b,m)
            batch_size=gt_boxes.shape[0]

            #---------- extract features from input
            fmaps=self.extractor(x)

            #---------- get proposal regions
            roi_boxes,roi_inds,anchors,rpn_pred_scores,rpn_pred_locs=self.rpn(fmaps,img_size,scale)

            #---------- RPN loss compute
            rpn_loss_locs=[]
            rpn_loss_scores=[]
            for b in range(batch_size):
                rpn_targets=self.anchor_target_creator(anchors,gt_boxes[b],img_size)
                rpn_loss=self.compute_rpn_loss(rpn_pred_scores[b],rpn_pred_locs[b],rpn_targets[1],rpn_targets[0])
                rpn_loss_locs.append(rpn_loss[0])
                rpn_loss_scores.append(rpn_loss[1])
            rpn_loss_locs=torch.stack(rpn_loss_locs,dim=0).mean()#()
            rpn_loss_scores=torch.stack(rpn_loss_scores,dim=0).mean()
            rpn_loss=[rpn_loss_locs,rpn_loss_scores]

            #---------- get rois and roi targets
            batch_sample_roi_boxes=[]
            batch_roi_target_locs=[]
            batch_roi_target_labels=[]
            batch_inds=[]
            for b in range(batch_size):
                sample_roi_boxes,roi_target_locs,roi_target_labels=self.proposal_target_creator(roi_boxes[roi_inds==b],
                                                                                                gt_boxes[b],
                                                                                                gt_labels[b])
                batch_sample_roi_boxes.append(sample_roi_boxes)
                batch_roi_target_locs.append(roi_target_locs)
                batch_roi_target_labels.append(roi_target_labels)
                index_b=b*torch.ones(len(sample_roi_boxes),device=sample_roi_boxes.device).long()
                batch_inds.append(index_b)
            batch_sample_roi_boxes=torch.cat(batch_sample_roi_boxes,dim=0)#[?,4]
            batch_roi_target_locs=torch.cat(batch_roi_target_locs,dim=0)#[?,4]
            batch_roi_target_labels=torch.cat(batch_roi_target_labels,dim=0)#[?,]
            batch_inds=torch.cat(batch_inds,dim=0)#[?,]

            #---------- get roi preds
            roi_pred_locs,roi_pred_scores=self.head(fmaps,batch_sample_roi_boxes,batch_inds)

            #---------- compute ROI loss
            roi_loss_locs=[]
            roi_loss_scores=[]
            for b in range(batch_size):
                roi_loss=self.compute_roi_loss(roi_pred_scores[batch_inds==b],roi_pred_locs[batch_inds==b],
                                                roi_target_labels[batch_inds==b],roi_target_locs[batch_inds==b])
                roi_loss_locs.append(roi_loss[0])
                roi_loss_scores.append(roi_loss[1])
            roi_loss_locs=torch.stack(roi_loss_locs,dim=0).mean()#()
            roi_loss_scores=torch.stack(roi_loss_scores,dim=0).mean()
            roi_loss=[roi_loss_locs,roi_loss_scores]
            
            return rpn_loss,roi_loss
            
        elif self.mode=="inference":
            pass

    
class AnchorTargetCreator():
    def __init__(self,
                sample_num=256,
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                pos_ratio=0.5):
        self.sample_num=sample_num
        self.pos_iou_thr=pos_iou_thr
        self.neg_iou_thr=neg_iou_thr
        self.pos_ratio=pos_ratio
    def __call__(self,anchors,gt_boxes,img_size):
        '''
        Args
        anchors [h*w*A,4] format (y1,x1,y2,x2)
        gt_boxes [m,4]
        img_size (2,) list
        '''
        # inside anchors
        inside_mask=(anchors[:,0]>=0)&(anchors[:,1]>0)&\
                                (anchors[:,2]<img_size[0])&(anchors[:,3]<img_size[1])#(h*w*A,)
        print("inside_mask",inside_mask.shape)
        inside_inds=torch.nonzero(inside_mask).squeeze(dim=-1)#(?,)
        print("inside_inds",inside_inds.shape)
        
        # labels: 1 is positive, 0 is negative, -1 is ignore
        labels_inside=torch.empty((len(inside_inds),),dtype=torch.int64,device=anchors.device)
        labels_inside.fill_(-1)#(?,)
        
        anchors_inside=anchors[inside_inds,:]#(?,4)
        ious=boxes_ious(anchors_inside,gt_boxes)#(?,m)
        max_ious=ious.max(dim=-1)[0]#(?,)
        print("anchors_inside",anchors_inside.shape,"ious",ious.shape,"max_ious",max_ious)
        labels_inside[max_ious<self.neg_iou_thr]=0
        labels_inside[max_ious>=self.pos_iou_thr]=1
        print("pos-lables",torch.nonzero(labels_inside==1).squeeze(dim=-1).shape)#(?,)

        a2g_maxiou_ind=ious.argmax(dim=1)#(?,)
        g2a_maxiou_ind=ious.argmax(dim=0)#(m,)
        print("a2g",a2g_maxiou_ind.shape,"g2a",g2a_maxiou_ind.shape)
        # according index for each anchor
        # maxiou_mask=torch.zeros_like(ious,dtype=torch.bool).scatter_(-1,maxiou_ind.unsqueeze(dim=-1),1)#(?,m)
        #if gtbox has no matching anchor, select the anchor which has max iou with it
        labels_inside[g2a_maxiou_ind]=1

        need_pos_num=int(self.pos_ratio*self.sample_num)
        pos_inds=torch.nonzero(labels_inside==1).squeeze(dim=-1)#(??,)
        pos_num=len(pos_inds)
        need_neg_num=int(self.sample_num-need_pos_num)
        neg_inds=torch.nonzero(labels_inside==0).squeeze(dim=-1)
        neg_num=len(neg_inds)
        print("need_pos/pos_num",need_pos_num,"/",pos_num,"need_neg/neg_num",need_neg_num,"/",neg_num)

        if pos_num>need_pos_num:
            dis_inds=pos_inds[random.sample(range(0,pos_num),pos_num-need_pos_num)]
            labels_inside[dis_inds]=-1
        if neg_num>need_neg_num:
            dis_inds=neg_inds[random.sample(range(0,neg_num),neg_num-need_neg_num)]
            labels_inside[dis_inds]=-1
        
        labels=torch.empty((anchors.shape[0],),dtype=torch.int64,device=anchors.device)
        labels.fill_(-1)#(?,)
        labels[inside_inds]=labels_inside##(h*w*A,)

        match_gt_inside=gt_boxes[a2g_maxiou_ind]#(?,4)
        locs_inside=anchor2loc(anchors_inside,match_gt_inside)#(?,4)

        locs=torch.empty((anchors.shape[0],4),dtype=torch.float32,device=anchors.device)
        locs.fill_(0)
        locs[inside_inds]=locs_inside

        return locs,labels
    
class RPNLOSS():
    '''
    for one batch
    '''
    def __call__(self,pred_scores,pred_locs,target_labels,target_locs):
        '''
        Args
        pred_locs [h*w*A,4]
        pred_scores [h*w*A,2]
        target_locs [h*w*A,4]
        target_labels [h*w*A]
        '''
        rpn_loss_scores=nn.functional.cross_entropy(pred_scores,target_labels,
                                                    ignore_index=-1)
        pos_target_locs=target_locs[target_labels==1]
        pos_pred_locs=pred_locs[target_labels==1]
        rpn_loss_locs=nn.functional.smooth_l1_loss(pos_pred_locs,pos_target_locs)
        return [rpn_loss_locs,rpn_loss_scores]
        
class ProposalTargetCreator():
    def __init__(self,
                sample_num=128,
                pos_iou_thr=0.5,
                neg_iou_thr_range=(0.0,0.5),
                pos_ratio=0.25,
                loc_mean=(0.,0.,0.),
                loc_std=(0.1,0.1,0.2,0.2)
                ):
        self.sample_num=sample_num
        self.pos_iou_thr=pos_iou_thr
        self.neg_iou_thr_range=neg_iou_thr_range
        self.pos_ratio=pos_ratio
        self.loc_mean=loc_mean
        self.loc_std=loc_std

    def __call__(self,roi_boxes,gt_boxes,gt_labels):
        '''
        roi_boxes [?,4]
        gt_boxes [m,4]
        gt_labels [m,]
        '''
        roi_boxes=torch.cat([roi_boxes,gt_boxes],dim=0)#[?,4]
        need_pos_num=int(self.pos_iou_thr*self.sample_num)
        need_neg_num=self.sample_num-need_pos_num
        ious=boxes_ious(roi_boxes,gt_boxes)#[?,m]
        max_ious=ious.max(dim=1)[0]#[?,]
        r2g_maxiou_ind=ious.argmax(dim=-1)#[?,]
        roi_target_labels=gt_labels[r2g_maxiou_ind]#[?,]
        
        pos_inds=torch.nonzero(max_ious>self.pos_iou_thr).squeeze(dim=-1)
        pos_num=len(pos_inds)
        neg_inds=torch.nonzero((max_ious>=self.neg_iou_thr_range[0])&(max_ious<self.neg_iou_thr_range[1])).squeeze(dim=-1)
        neg_num=len(neg_inds)
        if pos_num>need_pos_num:
            pos_inds=pos_inds[random.sample(range(0,pos_num),need_pos_num)]
        if neg_num>need_neg_num:
            neg_inds=neg_inds[random.sample(range(0,neg_num),need_neg_num)]
        
        keep_inds=torch.cat([pos_inds,neg_inds],dim=0)
        roi_target_labels=roi_target_labels[keep_inds]
        roi_target_labels[len(neg_inds):]=0
        sample_roi_boxes=roi_boxes[keep_inds]

        roi_target_locs=anchor2loc(sample_roi_boxes,gt_boxes[r2g_maxiou_ind[keep_inds]])#[??,4]
        roi_target_locs=(roi_target_locs-torch.tensor(self.loc_mean,dtype=roi_target_locs.dtype,device=roi_target_locs.device))\
                        /torch.tensor(self.loc_std,dtype=roi_target_locs.dtype,device=roi_target_locs.device)
        return sample_roi_boxes,roi_target_locs,roi_target_labels

class ROILOSS():
    def __call__(self,pred_scores,pred_locs,target_labels,target_locs):
        '''
        Args
        pred_scores [?,class_num+1]
        pred_locs [?,4]
        target_labels [?,]
        target_locs [?,4]
        '''
        roi_loss_scores=nn.functional.cross_entropy(pred_scores,target_labels)
        pos_target_locs=target_locs[target_labels>0]
        pos_pred_locs=pred_locs[target_labels>0]
        roi_loss_locs=nn.functional.smooth_l1_loss(pos_pred_locs,pos_target_locs)
        return [roi_loss_locs,roi_loss_scores]



def __test__():
    from rpn import gen_base_anchors,shift_anchors
    import cv2
    import numpy as np

    base_anchors=gen_base_anchors()
    anchors=shift_anchors(base_anchors,16,(38,63))
    print("base_anchors",base_anchors.shape,"anchors",anchors.shape)
    gt_boxes=torch.tensor([[100.,100.,500.,500.],[20.,20.,50.,100.],[150.,400.,400.,800.]]).float()
    a=AnchorTargetCreator(pos_iou_thr=0.5)(anchors,gt_boxes,(600,1000))
    print("locs_target",a[0].shape,"labels_target",a[1].shape)
    labels=a[1]
    pos_anchors=anchors[labels==1].numpy()
    print("pos-lables",pos_anchors.shape)#(?,)
    # draw=np.zeros((600,1000,3),dtype=np.uint8)
    # cv2.rectangle(draw,(100,100),(500,500),(0,0,255),3)
    # cv2.rectangle(draw,(20,20),(100,50),(0,0,255),3)
    # cv2.rectangle(draw,(400,150),(800,400),(0,0,255),3)
    
    # for ind,anchor in enumerate(pos_anchors):
    #     y1,x1,y2,x2=anchor
    #     pt1=(int(x1),int(y1))
    #     pt2=(int(x2),int(y2))
    #     ct=(int((x2+x1)/2),int((y2+y1)/2))
    #     cv2.rectangle(draw,pt1,pt2,(0,255,0),1)
    #     cv2.circle(draw,ct,4,(0,0,255),-1)
    # cv2.imwrite("pos_anchors.jpg",draw)
    
    # anchors=anchors.numpy()
    # anchors=np.reshape(anchors,(-1,9,4))
    # anchors=np.transpose(anchors,(1,0,2))
    # print(anchors.shape)
    # colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(200,20,100),(25,100,30)]
    # for ind,anchors in enumerate(anchors):
    #     color=colors[ind]
    #     for anchor in anchors[136:137]:
    #         y1,x1,y2,x2=anchor[:]
    #         pt1=(int(x1),int(y1))
    #         pt2=(int(x2),int(y2))
    #         cv2.rectangle(draw,pt1,pt2,color,1)
    # cv2.imwrite("anchors.jpg",draw)

if __name__ == "__main__":
    __test__()


        
