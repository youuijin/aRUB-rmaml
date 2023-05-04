import torch
import numpy as np
from    torch.nn import functional as F
from functorch import jacrev, vmap
from    learner import Learner

class aRUB_ja:
    def __init__(self, rho, q, n_way, k_qry, imgc, imgsz):
        self.rho = rho
        self.q = q
        self.n_way = n_way
        self.k_qry = k_qry
        self.imgc = imgc
        self.imgsz = imgsz
        self.device = torch.device('cuda:0')

    def norm_func(self, x):
            return torch.norm(x, p=self.q)
        
    def aRUBattack(self, data, label, net, weights):
        '''
        Computes approximation of logits
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        '''
        data.requires_grad = True
        net.zero_grad()
        logits = net(data, weights)
        
        logits = torch.sum(logits, dim=0)        
        flag = True
        for logit in logits:
            logit.backward(retain_graph=True)
            data_grad = torch.unsqueeze(data.grad, 1) #[75,3,28,28]->[75,1,3,28,28]
            if(flag):
                jacobian = data_grad.clone().detach()
                flag = False
            else:
                jacobian = torch.cat([jacobian,data_grad], dim = 1)
                
        for i in range(4):
            jacobian[:,4-i] = jacobian[:,4-i]-jacobian[:,3-i]


        #data = data.clone().detach()
        '''
        logits = net(data, weights)
        jacobian = torch.squeeze(vmap(jacrev(net))(torch.unsqueeze(data,1)))
        logits_label = []
        jac_label = []
        for cls in range(self.n_way):
            logits_label_cls = logits[cls*self.k_qry:(cls+1)*self.k_qry, label[cls*self.k_qry]]
            logits_label.append(logits_label_cls.tolist())
            jac_label_cls = jacobian[cls*self.k_qry:(cls+1)*self.k_qry, label[cls*self.k_qry]]
            jac_label.append(jac_label_cls.tolist())
        logits_label = torch.tensor(logits_label).view(-1).to(self.device)
        logits = logits - torch.unsqueeze(logits_label,1)
        jac_label = torch.tensor(jac_label).view(-1, self.imgc, self.imgsz, self.imgsz).to(self.device)
        jacobian = jacobian - torch.unsqueeze(jac_label, 1)
        jacobian = jacobian.view(-1, self.imgc, self.imgsz, self.imgsz)
        
        jac_norm_bat = vmap(self.norm_func)(jacobian)
        logits_adv = logits + self.rho * (jac_norm_bat.view(-1, self.n_way))

        return logits_adv