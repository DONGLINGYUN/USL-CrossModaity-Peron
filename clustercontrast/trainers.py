from __future__ import print_function, absolute_import
import time

from loss.loss import DCL, TripletLoss_ADP
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F

from loss.loss import SpatialAlignLoss


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
class CTripletLoss(nn.Module):
    def __init__(self, k_size=8, margin=0):
        super(CTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)

        dist_pc = (inputs - centers) ** 2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean())
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss, dist_pc.mean(), dist_an.mean()

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss
class ClusterContrastTrainer_pretrain(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data(inputs_ir)
            inputs_rgb, labels_rgb, indexes_rgb = self._parse_data(inputs_rgb)
            # forward
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            # f_out_rgb = self._forward(inputs_rgb)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)

            # loss_tri_rgb, batch_acc = self.tri(f_out_rgb, labels_rgb)
            # loss_tri_ir, batch_acc = self.tri(f_out_ir, labels_ir)
            # loss_tri = loss_tri_rgb+loss_tri_ir
            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + loss_tri
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir+loss_rgb#+loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

class ClusterContrastTrainer_ADCA_joint(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_ADCA_joint, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.cc_loss = CTripletLoss(k_size=16, margin=0.3).cuda()
        self.lable_loss = CrossEntropyLabelSmooth(num_classes=62)
        self.dcn  = DCL(num_pos=8, feat_norm='no')
        self.id = nn.CrossEntropyLoss().cuda()
        self.spatial = SpatialAlignLoss(mode='pos_neg')
        # self.tri = OriTripletLoss(64, 0.3)
        #self.tri = TripletLoss_WRT()
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir, data_loader_rgb, train_labelloader_ir, train_labelloader_rgb, optimizer,
                  print_freq=10, train_iters=400,i2r=None, r2i=None,cr2i=None):

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        correct = 0
        total = 0
        ##************---------***************

        #-------------------------------------
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()

            # add label data
            inputs_labelir = train_labelloader_ir.next()
            inputs_labelrgb = train_labelloader_rgb.next()
            # *---------------
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # ---------------label data-------
            inputs_labelir, tlabels_ir, indexes_tir = self._parse_data_ir(inputs_labelir)
            inputs_labelrgb, inputs_labelrgb1, tlabels_rgb, indexes_trgb = self._parse_data_rgb(inputs_labelrgb)
            # -------------------------------

            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            inputs_labelrgb = torch.cat((inputs_labelrgb,inputs_labelrgb1),0)
            tlabels_rgb = torch.cat((tlabels_rgb,tlabels_rgb),-1)
            # ---------------label data-------
            # 把有标记数据拼接到无标记数据上面


            inputs_rgb = torch.cat((inputs_labelrgb, inputs_rgb), 0)
            # labels_rgb = torch.cat((tlabels_rgb, labels_rgb), -1)
            inputs_ir = torch.cat((inputs_labelir,inputs_ir),0)
            # labels_ir = torch.cat((tlabels_ir, labels_ir), -1)
            targets = torch.cat((tlabels_rgb,tlabels_ir),0)

            # -------------------------------
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir,cls_score,label_data = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            #The class center losses of static and dynamic memery banks are calculated based on the class centers of the memory banks of the pseudo-label indexes ir and visible.
            #L_dyn and L_sta calculate the final sum according to the loss_ir and loss_rgb respectively.
            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            ###**********---------loss--------------------
            # loss_ll,_,_ = self.cc_loss(cls_score,targets)
            #loss_ll = self.lable_loss(cls_score,targets)
            # loss_ll = self.dcn(cls_score,targets)
            #Calculate the loss of labeled data
            loss_ll = self.id(cls_score,targets)
            loss_tri,batch_acc = self.tri(label_data,targets)
            correct += (batch_acc / 2)
            _, predicted = cls_score.max(1)
            correct += (predicted.eq(targets).sum().item() / 2)
            bat = int(labels_rgb.size(0) / 2)

            ##-#-------------------------------------------
            # Calculate class center losses based on mapping data
            if r2i:
                rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
                ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()
                #cr2i loss
                c_r2i = None
                c_i2r = None
                # k1=[]
                # f_rgbs = []
                #
                # for key1,f_rgb in zip(labels_rgb,f_out_rgb):
                #     if key1 in cr2i.keys():
                #         v = [v for k, v in cr2i.items() if k == key1]
                #         f_rgbs.append(f_rgb)
                #         k1.append(v[0])
                # if len(k1):
                #     f_rgbs = torch.stack(f_rgbs).cuda()
                #     c_r2i = torch.tensor(k1).cuda()
                #     c_r2iloss = self.memory_ir(f_rgbs, c_r2i.long())
                # else:
                #     c_r2iloss = torch.tensor(0.0)
                # k2= []
                # f_irs = []
                # for key2,f_ir in zip(labels_ir,f_out_ir):
                #     if key2 in cr2i.values():
                #         f_irs.append(f_ir)
                #         k = [k for k, v in cr2i.items() if v == key2]
                #         k2.append(int(k[0]))
                # if len(k2):
                #     f_irs = torch.stack(f_irs).cuda()
                #     c_i2r = torch.tensor(k2).cuda()
                #     c_i2rloss = self.memory_rgb(f_irs, c_i2r.long())
                # else:
                #     c_i2rloss = torch.tensor(0.0)
                #
                # c_loss = c_i2rloss + c_r2iloss



                alternate = True
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = 1 * self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    else:
                        cross_loss = 1 * self.memory_ir(f_out_rgb, rgb2ir_labels.long())
                else:
                    cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long()) + self.memory_ir(f_out_rgb,
                                                                                                  rgb2ir_labels.long())
                    # Unidirectional
                    # cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    # cross_loss = self.memory_ir(f_out_rgb, rgb2ir_labels.long())
            else:
                cross_loss = torch.tensor(0.0)
                c_loss = torch.tensor(0.0)
            
            loss = loss_ir+loss_rgb+loss_ll+0.25*cross_loss# + 0.25*c_loss+loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            total += targets.size(0)
            accu = 100. * correct / total
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss ll {:.3f}\t'
                      #'cross_loss {:.3f}\t'
                      #'loss_tri {:.3f}\t'
                      #'accu {:.3f}\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_ll))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct
