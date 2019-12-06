'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-12-05
@Email: xxxmy@foxmail.com
'''

import torch
import torch.nn as nn
from .roi_pooling import ROIPooling2D

class VGG16Head(nn.Module):
    def __init__(self,
                class_num,
                roi_size,
                feature_stride,
                classifier):
        super().__init__()
        self.class_num=class_num
        self.roi_size=roi_size
        self.feature_stride_scale=1./feature_stride
        self.classifier=classifier

        self.roi_locs=nn.Linear(4096,(class_num+1)*4)
        self.roi_scores=nn.Linear(4096,class_num+1)
        self.init_normal(self.roi_locs,0.,0.001)
        self.init_normal(self.roi_scores,0.,0.01)

        self.pooler=ROIPooling2D(roi_size,roi_size,self.feature_stride_scale)

    def init_normal(self,module,mean,std):
        
        nn.init.uniform_(module.weight,mean,std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.)

    def forward(self,fmaps,batch_sample_roi_boxes,batch_inds):
        '''
        fmaps [b,c,h,w]
        batch_sample_roi_boxes [?,4]
        batch_inds [?,]
        '''
        batch_size=fmap.shape[0]
        pool_f=[]
        for b in range(batch_size):
            b_pool_f=self.pooler(fmaps[b],batch_sample_roi_boxes[batch_inds==b])#[?,c*7*7]
            pool_f.append(b_pool_f)
        pool_f=torch.cat(pool_f,dim=0)
        fc7=self.classifier(pool_f)
        roi_pred_locs=self.roi_locs(fc7)
        roi_pred_scores=self.roi_scores(fc7)
        return roi_pred_locs,roi_pred_scores
