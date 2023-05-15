#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from torch.autograd import Variable
from    learner import Learner
from    copy import deepcopy
#from attack import PGD
from deepfoolattack import deepfool
from aRUBattack import aRUB
import time

from functorch import jacrev, vmap


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, device):
        """
        :param args:
        """
        print("init")
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.adv_lr = args.adv_lr
        self.rho = args.rho
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.device = device
        self.imgc = args.imgc
        self.imgsz = args.imgsz
        self.args = args

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        #self.meta_optimadv = optim.Adam(self.netadv.parameters(), lr=self.meta_lr)
        self.meta_optim_adv = optim.Adam(self.net.parameters(), lr=self.adv_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        make_time = 0 # adv sample 생성 시간
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        
        need_adv = True
        #AT
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        losses_q_adv = [0 for _ in range(self.update_step + 1)]
        corrects_adv = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):
            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct
                

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct
                

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] = losses_q[k+1] + loss_q
                
                #aRUB attack
                if need_adv and k == self.update_step - 1: # for meta-update
                    
                    net_copy = deepcopy(self.net)
                    
                    at= aRUB(rho=self.rho, q=1, n_way=self.n_way, k_qry=self.k_qry, imgc=self.imgc, imgsz=self.imgsz)
                    optimizer.zero_grad()
                    
                    t = time.perf_counter()
                    logits_q_adv = at.aRUBattack(data=x_qry[i], label=y_qry[i], net=net_copy, weights=fast_weights)
                    make_time += time.perf_counter() - t
                    
                    loss_q_adv = F.cross_entropy(logits_q_adv, y_qry[i])
                    losses_q_adv[k + 1] = losses_q_adv[k+1] + loss_q_adv
                
                #logits_q = self.net(x_qry[i], fast_weights, bn_training=True) -> 위로 올림
                #loss_q will be overwritten and just keep the loss_q on last update step.

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
                    
                    #PGD AT
                    if need_adv and k == self.update_step - 1:
                        pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                        correct_adv = torch.eq(pred_q_adv, y_qry[i]).sum().item()
                        corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv

                

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num #마지막 원소 사용 -> K step update한 후!
        
        loss_q_adv = losses_q_adv[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        
        if need_adv:
            self.meta_optim_adv.zero_grad()
            loss_q_adv.backward()
            self.meta_optim_adv.step()
            accs_adv = np.array(corrects_adv) / (querysz * task_num)
        else:
            accs_adv = 0
        

        return accs, accs_adv, loss_q, loss_q_adv, make_time


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        
        need_adv = True
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.update_lr, momentum=0.9, weight_decay=5e-4)
        eps, step = (2.0,10)
        corrects_adv = [0 for _ in range(self.update_step_test + 1)]
        corrects_adv_prior = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
          
        #DeepFool AT
        if need_adv:
            at = deepfool(num_classes = self.n_way, device = self.device)
            
            optimizer.zero_grad()
            '''
            adv_inp = torch.zeros(x_qry.shape).to(self.device)
            for i in range(x_qry.shape[0]):
                data = x_qry[i]
                label = y_qry[i]
                adv_inp_adv_i = at.dfattack(net, data, label)
                adv_inp[i] = adv_inp_adv_i
                '''
            adv_inp = at.dfattack(net, x_qry, y_qry)
        

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            #find the correct index
            corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
            
        
        #DeepFool AT
        if need_adv:
            optimizer.zero_grad()
            '''
            adv_inp = torch.zeros(x_qry.shape).to(self.device)
            for i in range(x_qry.shape[0]):
                data = x_qry[i]
                label = y_qry[i]
                adv_inp_adv_i = at.dfattack(net, data, label)
                adv_inp[i] = adv_inp_adv_i
                '''
            adv_inp = at.dfattack(net, x_qry, y_qry)
            with torch.no_grad():
                logits_q_adv = net(adv_inp, net.parameters(), bn_training=True)
                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, y_qry).sum().item()
                correct_adv_prior = torch.eq(pred_q_adv[corr_ind], y_qry[corr_ind]).sum().item()
                corrects_adv[0] = corrects_adv[0] + correct_adv
                if len(corr_ind)!=0:
                    corrects_adv_prior[0] = corrects_adv_prior[0] + correct_adv_prior/len(corr_ind)
                ########수정한 부분##########

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            #find the correct index
            corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
            
            
            #PGD AT
            if need_adv:
                logits_q_adv = net(adv_inp, fast_weights, bn_training=True)
                pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                correct_adv = torch.eq(pred_q_adv, y_qry).sum().item()
                correct_adv_prior = torch.eq(pred_q_adv[corr_ind], y_qry[corr_ind]).sum().item()
                corrects_adv[1] = corrects_adv[1] + correct_adv
                corrects_adv_prior[1] = corrects_adv_prior[1] + correct_adv_prior/len(corr_ind)
            

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            
            #DeepFool AT
            
            if need_adv:
                at = deepfool(num_classes = self.n_way, device = self.device)
                data = x_qry
                label = y_qry
                optimizer.zero_grad()
                '''
                adv_inp = torch.zeros(x_qry.shape).to(self.device)
                for i in range(x_qry.shape[0]):
                    data = x_qry[i]
                    label = y_qry[i]
                    adv_inp_adv_i = at.dfattack(net, data, label)
                    adv_inp[i] = adv_inp_adv_i
                    '''
                adv_inp = at.dfattack(net, x_qry, y_qry)

                logits_q_adv = net(adv_inp, fast_weights, bn_training=True)
                loss_q_adv = F.cross_entropy(logits_q_adv, y_qry)
            
            

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                #find the correct index
                corr_ind = (torch.eq(pred_q, y_qry) == True).nonzero()
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
                
                
                #PGD AT
                if need_adv:
                    pred_q_adv = F.softmax(logits_q_adv, dim=1).argmax(dim=1)
                    correct_adv = torch.eq(pred_q_adv, y_qry).sum().item()
                    correct_adv_prior = torch.eq(pred_q_adv[corr_ind], y_qry[corr_ind]).sum().item()
                    corrects_adv[k + 1] = corrects_adv[k + 1] + correct_adv
                    corrects_adv_prior[k + 1] = corrects_adv_prior[k + 1] + correct_adv_prior/len(corr_ind)


        del net

        accs = np.array(corrects) / querysz
        
        accs_adv = np.array(corrects_adv) / querysz
        
        accs_adv_prior = np.array(corrects_adv_prior)

        return accs, accs_adv, accs_adv_prior, loss_q, loss_q_adv


def main():
    pass


if __name__ == '__main__':
    main()

