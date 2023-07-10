import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  argparse
from    metaTrain import Meta
import  sys

from torch.utils.tensorboard import SummaryWriter

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0] 
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def main(args):
    sum_str_path = "./runs_size/"
    sum_str = str(args.imgsz)+"/"+str(args.attack)+"/" +str(args.test_attack)+ "_" +str(args.meta_lr) +"_"+str(args.adv_lr)
    
    writer = SummaryWriter(sum_str_path+sum_str, comment=sum_str)
    print("Save Path: "+sum_str_path+sum_str)
    print("Training attack: " + args.attack + ", Test attack: " + args.test_attack)
    
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    s = (args.imgsz-2)//2
    s = (s-2)//2
    s = s-3

    config = [
        ('conv2d', [32, 3, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [32, 32, 3, 3, 1, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d', [2, 1, 0]),
        ('flatten', []),
        ('linear', [args.n_way, 32 * s * s])
    ]
    
    device = torch.device('cuda:'+str(args.device_num))
    maml = Meta(args, config, device).to(device)

    #start_epoch = 0
    #start_step = 0
    #filename = 'mamlfgsmeps4_2.pt'
    #filename = 'mamlfgsmeps2_8.pt'
    #maml = Meta(args, config).to(device)
    # if os.path.isfile(filename):
    #     print("=> loading checkpoint '{}'".format(filename))
    #     checkpoint = torch.load(filename)
    #     start_epoch = checkpoint['epoch']
    #     start_step = checkpoint['step']
    #     maml.net.load_state_dict(checkpoint['state_dict'])
    #     #maml = maml.to(device)
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(filename, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(filename))
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    # batchsz here means total episode number
    mini = MiniImagenet('../../dataset', mode='train', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('../../dataset', mode='test', n_way=args.n_way, k_shot=args.k_spt,
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)

    #ExecTime = 0 # training 시간
    #SampleTime = 0 # training 중 adv sample 생성 시간
    tot_step = 0
    for epoch in range(args.epoch):
        # fetch meta_batchsz num of episode each time
        #t = time.perf_counter()
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):
            tot_step = tot_step + args.task_num

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)
            accs, accs_adv, loss_q, loss_q_adv, make_time = maml(x_spt, y_spt, x_qry, y_qry)
            #SampleTime += make_time
            if step % 20 == 0:
                print('step:', step, '\ttraining acc:', accs)
                print('step:', step, '\ttraining acc_adv:', accs_adv)
                writer.add_scalar("acc/train", accs[-1], tot_step)
                writer.add_scalar("acc_adv/train", accs_adv[-1], tot_step)
                writer.add_scalar("loss/train", loss_q, tot_step)
                writer.add_scalar("loss_adv/train", loss_q_adv, tot_step)
                state = {'epoch': epoch, 'step': step, 'state_dict': maml.net.state_dict()}
                torch.save(state, 'mamlfgsmeps4_2.pt')
            
        # evaluation -> 학습에는 전혀 영향을 주지 않음, copy network를 사용하므로
        #ExecTime += time.perf_counter() - t
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True)
        accs_all_test = []
        accsadv_all_test = []
        accsadvpr_all_test = []
        loss_all_test = []
        loss_adv_all_test = []
        
        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                            x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, accs_adv, accs_adv_prior, loss_q, loss_q_adv = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)
            accsadv_all_test.append(accs_adv)
            accsadvpr_all_test.append(accs_adv_prior)
            loss_all_test.append(loss_q.item())
            loss_adv_all_test.append(loss_q_adv.item())
            
        # [b, update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        accs_adv = np.array(accsadv_all_test).mean(axis=0).astype(np.float16)
        accs_adv_prior = np.array(accsadvpr_all_test).mean(axis=0).astype(np.float16)
        loss_q = np.array(loss_all_test).mean()
        loss_q_adv = np.array(loss_adv_all_test).mean()
        print('Test acc:', accs)
        print('Test acc_adv:', accs_adv)
        print('Test acc_adv_prior:', accs_adv_prior)
        
        writer.add_scalar("acc/test_epoch", accs[-1],epoch)
        writer.add_scalar("acc_adv/test_epoch", accs_adv[-1],epoch)
        writer.add_scalar("loss/epoch", loss_q, epoch)
        writer.add_scalar("loss_adv/epoch", loss_q_adv, epoch)
        
        #writer.add_scalar("train_time/epoch", ExecTime, epoch)
        #writer.add_scalar("make_time/epoch", SampleTime, epoch)
            

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # Meta-learning options
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)

    # Dataset options
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    
    # Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001) #0.001 - 0.0002 기존
    argparser.add_argument('--adv_lr', type=float, help='adv-level learning rate', default=0.0008)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--device_num', type=int, help='what gpu to use', default=0)

    # adversarial attack options
    argparser.add_argument('--attack', type=str, default="aRUB")
    argparser.add_argument('--test_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--eps', type=float, help='attack-eps', default=0.3)
    argparser.add_argument('--rho', type=float, help='aRUB-rho', default=0.01)

    args = argparser.parse_args()

    main(args)