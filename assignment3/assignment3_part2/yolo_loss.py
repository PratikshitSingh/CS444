
import torch

from torch.autograd import Variable
from multiprocessing import reduction

#
import torch.nn as nn
import torch.nn.functional as F

def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]
    

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        new_boxes = torch.zeros(boxes.size()).cuda() # to be checked
        
        #for i in range(boxes.size()[0]):
        new_boxes[:,0], new_boxes[:,1] = boxes[:,0]/self.S - 0.5*boxes[:,2], boxes[:,1]/self.S - 0.5*boxes[:,3]
        new_boxes[:,2], new_boxes[:,3] = boxes[:,0]/self.S + 0.5*boxes[:,2], boxes[:,1]/self.S + 0.5*boxes[:,3]
        return new_boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        pred_box_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        

        ### CODE ###
        # Your code here
        

        ious_1 =torch.diag( compute_iou(self.xywh2xyxy(pred_box_list[0][:,:4]),self.xywh2xyxy(box_target)) , 0 )
        
        ious_2=torch.diag(compute_iou(self.xywh2xyxy(pred_box_list[1][:,:4]),self.xywh2xyxy(box_target)), 0)

 	
 	#initializing best ious and best boxes class
        best_ious = torch.zeros((pred_box_list[0].size()[0],1))
        best_ious = best_ious.cuda()
        best_boxes = torch.zeros_like(pred_box_list[0])
        best_boxes = best_boxes.cuda()
        
        for nss in range( ious_1.shape[0] ):
            if ious_1[nss] > ious_2[nss]:
                best_ious[nss ,0] = ious_1[nss]
                best_boxes[nss,:5] = pred_box_list[0][nss,:5]
            else:
                best_ious[nss,0] = ious_2[nss]
                best_boxes[nss,:5] = pred_box_list[1][nss,:5]
        return best_ious, best_boxes

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        

        Returns:
        class_loss : scala
        """
        ### CODE ####
        # Your code here
        has_object_map = has_object_map.unsqueeze(-1).expand(classes_pred.shape[0],self.S,self.S,20)
        
        return F.mse_loss(classes_pred[has_object_map], classes_target[has_object_map], reduction='sum')

    def get_no_object_loss(self, pred_boxes_list, has_object_map):

        """

        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        #transforming the no_object_map
        no_object_map=torch.ones_like(has_object_map).cuda() ^ has_object_map
        no_object_map= no_object_map.unsqueeze(3).expand( pred_boxes_list[0].size(0) , self.S , self.S,5)
        # for each of the bounding box
        no_object_pred1 = pred_boxes_list[0][no_object_map].reshape(-1,5)
        no_object_pred2 = pred_boxes_list[1][no_object_map].reshape(-1,5)

	# concatenating the tensors
        preds=torch.cat((no_object_pred1[:,4],no_object_pred2[:,4]),0)

        return self.l_noobj * F.mse_loss(preds,torch.zeros_like(preds).cuda(),reduction='sum')

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient
        
       
        """
        ### CODE
        # your code here
        loss_fn = F.mse_loss
        box_target_conf=box_target_conf.detach()
        contain_loss = loss_fn(box_pred_conf, box_target_conf, reduction = 'sum')
        return contain_loss
        

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODEnse[:,2:])
        
        loss = F.mse_loss( box_pred_response[:,:2] ,box_target_response[:,:2],reduction='sum')
        loss += F.mse_loss(torch.sqrt(box_pred_response[:,2:]),torch.sqrt(box_target_response[:,2:]),reduction='sum')
        
        return loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        pred_boxes_list1 = pred_tensor[:, :, : , : 5]
        pred_boxes_list2 = pred_tensor[:, :, : , 5: 10]
        pred_boxes_list = [pred_boxes_list1 , pred_boxes_list2] # [(N,S,S,5), (N,S,S,5) ...B times]
        
        pred_cls = pred_tensor[:, :, :, 10:] # N,S,S,20 (C)


        # compcute classification loss
        cls_loss=self.get_class_prediction_loss(pred_cls,target_cls,has_object_map)


        # compute no-object loss
        no_obj_loss =  self.get_no_object_loss(pred_boxes_list,has_object_map)



        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation

        has_object_map_2 = has_object_map.unsqueeze(3).expand(N , self.S , self.S , 5)
        pred_boxes_list_hs=[pred_boxes_list[0][has_object_map_2].contiguous().view(-1,5),pred_boxes_list[1][has_object_map_2].contiguous().view(-1,5)]

        # doing it the same way as in the previous 
        has_object_map_3=has_object_map.unsqueeze(3).expand(N,self.S,self.S,4)
        target_boxes_hs=target_boxes[has_object_map_3].contiguous().view(-1,4)
	#print('sum of has objet map :', torch.sum(has_object_map_2))
        #print('target boxes shape ::' , target_boxes.size())
        #print('target boxes shape ::' , pred_boxes_list[0].size())
        
        best_iou, best_bound_box=self.find_best_iou_boxes(pred_boxes_list_hs,target_boxes_hs)



        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        reg_loss=self.l_coord *(self.get_regression_loss(best_bound_box[:,:4],target_boxes_hs))

        # compute contain_object_loss here

 

        contain_obj_loss = self.get_contain_conf_loss(best_bound_box[:,4].reshape(-1,1),best_iou)


        # compute final loss

        total_loss=cls_loss +no_obj_loss+contain_obj_loss + reg_loss



        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss/N,
            reg_loss=reg_loss/N,
            containing_obj_loss=contain_obj_loss/N,
            no_obj_loss=no_obj_loss/N,
            cls_loss=cls_loss/N,
        )
        return loss_dict

