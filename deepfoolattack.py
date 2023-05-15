'''
import numpy as np
import copy

class deepfool:
    def __init__(self, num_classes=5, overshoot=0.02, max_iter=10, device='cuda:0'):
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.device = device
    
    #not batched
    def dfattack(self, net, image, label):
        """
           :param image: Image of size HxWx3
           :param net: network (input: images, output: values of activation **BEFORE** softmax).
           :param num_classes: num_classes (limits the number of classes to test against, by default = 5)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
        """
        input_shape = image.shape
        pert_image = copy.deepcopy(image) #x_0
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0
            
        x = pert_image
        x.requires_grad = True

        fs_list = net(x).squeeze()
        k_i = label

        while k_i == label and loop_i < self.max_iter:
            pert = np.inf
            fs_list[label].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy() #grad f_k(x_i)

            for k in range(0, self.num_classes):
                if k==label:
                    continue
                x.grad.zero_()

                fs_list[k].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy() #grad f_k(x_i)

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs_list[k]-fs_list[label]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k #find minimum of perturbation
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            pert_image = image + (1+self.overshoot)*torch.from_numpy(r_tot).to(self.device)

            x = pert_image
            x.requires_grad = True
            fs_list = net(x).squeeze()
            k_i = fs_list.argmax().cpu().numpy()

            loop_i += 1

        r_tot = (1+self.overshoot)*r_tot

        return pert_image

    '''
import numpy as np
import copy
import torch
import functorch

class deepfool:
    def __init__(self, num_classes=5, overshoot=0.02, max_iter=10, device='cuda:0'):
        self.num_classes = num_classes
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.device = device
    
    def dfattack(self, net, images, labels):
        """
           :param images: batch of images of size BxHxWx3
           :param net: network (input: images, output: values of activation **BEFORE** softmax).
           :param num_classes: num_classes (limits the number of classes to test against, by default = 5)
           :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
           :param max_iter: maximum number of iterations for deepfool (default = 50)
           :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed images
        """
        input_shape = images.shape
        pert_images = copy.deepcopy(images) # x_0
        w = torch.zeros(input_shape, device=self.device)
        r_tot = torch.zeros(input_shape, device=self.device)

        loop_i = 0

        x = pert_images.clone()
        x.requires_grad = True

        fs_list = net(x).squeeze()
        k_i = labels.clone()

        while (k_i == labels).sum() > 0 and loop_i < self.max_iter:
            pert = torch.full((labels.shape[0],), np.inf, device=self.device) #[75] -> 각 pert의 최소를 구하기 위함
            fs_list[labels].backward(torch.ones_like(fs_list[labels]), retain_graph=True)
            grad_orig = x.grad.data.clone() # grad f_k(x_i)

            for k in range(0, self.num_classes):
                mask = (k != labels)
                if mask.sum() > 0:
                    x.grad.zero_()

                    fs_list[k].backward(torch.ones_like(fs_list[k]), retain_graph=True)
                    cur_grad = x.grad.data.clone() # grad f_k(x_i)
                    
                    # set new w_k and new f_k
                    w_k = cur_grad - grad_orig #[75,3,28,28]
                    fs_label = torch.zeros([75]).to(self.device)
                    for i in range(self.num_classes):
                        fs_label[i*15:(i+1)*15] = fs_list[i*15,labels[i*15]].data
                    f_k = (fs_list[:,k] - fs_label).data #[75,1]

                    #print(torch.norm(w_k.reshape(w_k.shape[0], -1), dim=1).shape)
                    #pert_k = torch.abs(f_k) / torch.norm(w_k.reshape(w_k.shape[0], -1), dim=1)
                    #pert_k = torch.vmap(lambda f, w: (torch.abs(f)/torch.norm(w.reshape(w.shape[0],-1, dim=1))), (f_k, w_k))
                    #pert_k = functorch.vmap(lambda f, w: torch.abs(f) / torch.norm(w.reshape(w.shape[0], -1), dim=1))(f_k, w_k)
                    #print(pert_k)
                    pert_k = torch.div(torch.abs(f_k), torch.norm(w_k.reshape(w_k.shape[0],-1), dim=1))
                    
                   # determine which w_k to use
                    #pert[mask] = torch.minimum(pert[mask], pert_k[mask])
                    pert[mask] = torch.minimum(pert[mask], pert_k[mask])
                    w[mask] = w_k[mask]

            # compute r_i and r_tot
            r_i = torch.div((pert+1e-4).view(75,1,1,1).expand(75,3,28,28) * w, torch.norm(w.reshape(w.shape[0], -1), dim=1, keepdim=True).view(75,1,1,1).expand(75,3,28,28))
            
            r_tot = r_tot + r_i

            pert_images = images + (1 + self.overshoot) * r_tot

            x = pert_images.clone()
            x.requires_grad = True
            fs_list = net(x).squeeze()
            k_i = fs_list.argmax(dim=1)

            loop_i += 1

        r_tot = (1 + self.overshoot) * r_tot

        return pert_images
