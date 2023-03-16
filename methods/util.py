

import os
import numpy as np
import torch
import json
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from configs.pan import pan_r18_fpem_v1

img_size = 32

def setup_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def set_lr(opts,lr):
    for op in opts :
        for param_group in op.param_groups:
            param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class WarmUpLrSchedule:

    def __init__(self, warm_epoch, epoch_tot_steps, init_lr):
        self.ep_steps = epoch_tot_steps
        self.tgtstep = warm_epoch * epoch_tot_steps
        self.init_lr = init_lr
        self.warm_epoch = warm_epoch

    def get_lr(self,epoch,step,lr):
        tstep = epoch * self.ep_steps + step
        if self.tgtstep > 0 and tstep <= self.tgtstep:
            lr = self.init_lr *  tstep / self.tgtstep
        return lr

class MultiStepLrSchedule:

    def __init__(self,milestones,lrdecays,start_lr,warmup_schedule = None):
        super().__init__()
        self.milestones = milestones
        self.warmup = warmup_schedule
        self.lrdecays = lrdecays
        self.start_lr = start_lr

    # step 表示epoch中已经输入过的样本数
    def get_lr(self,epoch,step,lr):
        lr = self.start_lr
        # if step == 0 : # update learning rate
        for m in self.milestones:
            if epoch >= m:
                lr *= self.lrdecays
        # print("LEARNRATE",lr)
        if self.warmup is not None:
            lr = self.warmup.get_lr(epoch,step,lr)
        # print("LEARNRATE",lr)
        return lr

# cosine_s,cosine_e = 0,0


# epoch wise
class EpochwiseCosineAnnealingLrSchedule:

    def __init__(self,startlr,milestones,lrdecay,epoch_num,warmup = None):
        super().__init__()
        self.cosine_s,self.cosine_e = 0,0
        self.milestones = milestones
        self.lrdecay = lrdecay
        self.warmup = warmup
        self.warmup_epoch = 0 if warmup is None else warmup.warm_epoch
        self.epoch_num = epoch_num
        self.startlr = startlr
        self.ms = [self.warmup_epoch] + self.milestones + [self.epoch_num]
        self.ref = {self.ms[i] : self.ms[i+1] for i in range(len(self.ms)-1)}

    def get_lr(self,epoch,step,lr):
        #global cosine_s,cosine_e
        if self.warmup is not None:
            lr = self.warmup.get_lr(epoch,step,lr)
        if step != 0 :
            return lr
        if epoch in self.ms:
            if epoch != self.warmup_epoch:
                self.startlr *= self.lrdecay
            self.cosine_s = epoch
            self.cosine_e = self.ref[epoch]
        #print("calc lr",epoch,self.ms,self.cosine_s,self.cosine_e)
        if self.cosine_e > 0:
            lr = self.startlr * (np.cos((epoch - self.cosine_s) / (self.cosine_e - self.cosine_s) * 3.14159)+1) * 0.5
        
        return lr

# Step wise
class StepwiseCosineAnnealingLrSchedule:

    def __init__(self,startlr,epoch_tot_steps,milestones,lrdecay,epoch_num,warmup = None):
        super().__init__()
        self.cosine_s,self.cosine_e = 0,0
        self.milestones = milestones
        self.lrdecay = lrdecay
        self.warmup = warmup
        self.warmup_epoch = 0 if warmup is None else warmup.warm_epoch
        self.epoch_num = epoch_num
        self.startlr = startlr
        self.ms = [self.warmup_epoch] + self.milestones + [self.epoch_num]
        self.ref = {self.ms[i] : self.ms[i+1] for i in range(len(self.ms)-1)}
        self.ep_steps = epoch_tot_steps

    # step wise
    def get_lr(self,epoch,step,lr):
        if self.warmup is not None:
            lr = self.warmup(epoch,step,lr)
        if step == 0 and epoch in self.ms:
            if epoch != self.warmup_epoch:
                self.startlr *= self.lrdecay
            self.cosine_s = epoch
            self.cosine_e = self.ref[epoch]
        if self.cosine_e > 0:
            steps = step + (epoch - self.cosine_s) * self.epoch_tot_steps
            lr = self.startlr  * (np.cos( steps / (self.cosine_e - self.cosine_s) / self.epoch_tot_steps * 3.14159)+1) * 0.5
        return lr

def get_scheduler(config,train_loader):
    if config['lr_schedule'] == 'multi_step':
        warmup = WarmUpLrSchedule(config['warmup_epoch'], len(train_loader) ,config['learn_rate'])
        return MultiStepLrSchedule(config["milestones"],config['lr_decay'],config['learn_rate'],warmup)
    elif config['lr_schedule'] == 'cosine':
        warmup = WarmUpLrSchedule(config['warmup_epoch'], len(train_loader) ,config['learn_rate'])
        return EpochwiseCosineAnnealingLrSchedule(config['learn_rate'],config["milestones"],config['lr_decay'],config['epoch_num'],warmup)

def visualize(feats, gts, pred, open_pred, config, args, saving_path):

    temp_x = []
    temp_y = []

    with open(args.ds,'r') as f:
        settings = json.load(f)
    
    model_name = config["category_model"]['model']

    feats = feats.view(len(feats), -1)
    feats = feats.cpu()

    plt.figure(figsize=(30, 10))

    print('visualizing features...')
    tsne = TSNE(n_components=2,random_state=0)
    cluster = tsne.fit_transform(feats)

    px = cluster[:,0]
    py = cluster[:,1]

    dataset_name = settings['name']
    # classes = ['0', '1', '2', '3', '4', '5']
    classes = settings['train'][0]['keep_class']
    # open_classes = ['0', '1', '2', '3', '4', '5', '-1']
    open_classes = settings['test'][0]['keep_class']
    colors = ['Red', 'BlueViolet', 'LawnGreen', 'HotPink', 'Orange', 'Chartreuse', 'Cyan', 'Khaki', 'Azure', 'BlanchedAlmond',
                'Aquamarine', 'BurlyWood', 'CadetBlue', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'DarkCyan',
                'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
                'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'Gold',
                'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'IndianRed', 'Ivory', 'Lavender', 'LavenderBlush', 'AliceBlue', 'MistyRose',
                'Moccasin', 'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff',
                'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple']

    plt.subplot(1,3,1) 
    plt.scatter(px, py, marker='.', label='unknown', c='black')
    for i in range(len(classes)):
        idx = [x for x in range(len(gts)) if gts[x]==i]
        for j in idx:
            temp_x.append(px[j])
            temp_y.append(py[j])
        plt.scatter(temp_x, temp_y, marker='.', label=classes[i], c=colors[i])
        temp_x.clear()
        temp_y.clear()
    plt.legend()
    plt.title(f'ground truth_{model_name}')
    
    plt.subplot(1,3,2)
    plt.scatter(px, py, marker='.', label='unknown', c='black')
    for i in range(len(classes)):
        idx = [x for x in range(len(pred)) if pred[x]==i]
        for j in idx:
            temp_x.append(px[j])
            temp_y.append(py[j])
        plt.scatter(temp_x, temp_y, marker='.', label=classes[i], c=colors[i])
        temp_x.clear()
        temp_y.clear()
    plt.legend()
    plt.title(f'predictions_{model_name}')

    plt.subplot(1,3,3)
    plt.scatter(px, py, marker='.', label='unknown', c='black')
    for i in range(len(classes)):
        idx = [x for x in range(len(open_pred)) if open_pred[x]==i]
        for j in idx:
            temp_x.append(px[j])
            temp_y.append(py[j])
        plt.scatter(temp_x, temp_y, marker='.', label=classes[i], c=colors[i])
        temp_x.clear()
        temp_y.clear()
    plt.legend()
    plt.title(f'openset recognition_{model_name}')

    plt.suptitle(f'feature_{dataset_name}_{model_name}', fontsize='x-large')
    plt.tight_layout()
    plt.savefig(f'./{saving_path}feature_{dataset_name}_{model_name}.png')
    print('visualization end')
    
def create_target_map(cls_list, num_classes):
    '''
    Creates a mapping from original dataset labels to new 'known class' training label
	known_classes: classes that will be trained with
	num_classes: number of classes the dataset typically has
		
	returns mapping - a dictionary where mapping[original_class_label] = known_class_label
    '''
    mapping = [None for i in range(num_classes)]
    
    # known_classes = cls_list
    cls_list = [x for x in range(len(cls_list)) if not cls_list[x]==-1]

    for i, num in enumerate(cls_list):
        mapping[num] = i
    
    return mapping

def find_anchor_means(net, dataloader, num_classes, only_correct = False):
    ''' Tests data and fits a multivariate gaussian to each class' logits. 
        If dataloaderFlip is not None, also test with flipped images. 
        Returns means and covariances for each class. '''
    #find gaussians for each class
    # if datasetName == 'MNIST' or datasetName == "SVHN":
    #     loader, _ = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
    #     logits, labels = gather_outputs(net, mapping, loader, only_correct = only_correct)
    # else:
    #     loader, loaderFlipped = dataHelper.get_anchor_loaders(datasetName, trial_num, cfg)
    #     logits, labels = gather_outputs(net, mapping, loader, loaderFlipped, only_correct = only_correct)
    
    logits, labels, pred = gather_outputs(net, dataloader, only_correct = only_correct)

    means = [None for i in range(num_classes)]

    for cl in range(num_classes):   
        x = logits[labels == cl]
        x = np.squeeze(x)
        means[cl] = np.mean(x)
    return means

def  gather_outputs(model, dataloader, output_feature=None, dataloaderFlip = None, data_idx = 0, 
                    calculate_scores = False, only_correct = False, args=None, datasets=None):
    ''' Tests data and returns outputs and their ground truth labels.
        data_idx        0 returns logits, 1 returns distances to anchors
        use_softmax     True to apply softmax
        unknown         True if an unknown dataset
        only_correct    True to filter for correct classifications as per logits
    '''
    X = []
    y = []
    prediction = []
    
    if args.use_neck:
        if datasets == "tinyimagenet":
            if pan_r18_fpem_v1.model['neck']['out_feature_map'] == 4:
                temp = torch.zeros(1,512,8,8).cuda()
            elif pan_r18_fpem_v1.model['neck']['out_feature_map'] == 8:
                temp = torch.zeros(1,512,16,16).cuda()
            else:
                raise Exception("output feature map size error")
        else:
            if pan_r18_fpem_v1.model['neck']['out_feature_map'] == 4:
                temp = torch.zeros(1,512,4,4).cuda()
            elif pan_r18_fpem_v1.model['neck']['out_feature_map'] == 8:
                temp = torch.zeros(1,512,8,8).cuda()
            else:
                raise Exception("output feature map size error")
    else:
        if datasets == "tinyimagenet":
            temp = torch.zeros(1,512,8,8).cuda()
        else:
            temp = torch.zeros(1,512,4,4).cuda()
        
    if calculate_scores:
        softmax = torch.nn.Softmax(dim = 1)

    for i, data in enumerate(dataloader):
        images, labels = data[0].cuda(), data[1].cuda()
        images = images.cuda()

        # if unknown:
        #     targets = labels
        # else:
        #     targets = mapping
        
        xcls, pred, feat = model(images, labels)   # ex) 1 x 2
        prediction.append(pred.detach().cpu())
        logits = xcls[0] # ex) 1x20
        distances = xcls[1]  # ex) 1x20
        temp = torch.cat([temp, feat], dim=0)

        if only_correct:
            if data_idx == 0:
                _, predicted = torch.max(logits, 1)
            else:
                _, predicted = torch.min(distances, 1)
            
            mask = predicted == labels
            logits = logits[mask]
            distances = distances[mask]
            labels = labels[mask]

        if calculate_scores:
            softmin = softmax(-distances)
            invScores = 1-softmin
            scores = distances*invScores
        else:
            if data_idx == 0:
                scores = logits
            if data_idx == 1:
                scores = distances

        X += scores.detach().cpu().tolist()
        y += labels.cpu().tolist()

    if dataloaderFlip is not None:
        for i, data in enumerate(dataloaderFlip):
            images, labels = data
            images = images.cuda()

            # if unknown:
            #     targets = labels
            # else:
            #     targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()
            
            pred,xcls,feat = model(images)
            prediction.append(pred.detach().cpu())
            logits = xcls[0]
            distances = xcls[1]
            temp = torch.cat([temp, feat], dim=0)

            if only_correct:
                if data_idx == 0:
                    logits, predicted = torch.max(logits, 1)
                else:
                    logits, predicted = torch.min(distances, 1)
                mask = predicted == labels
                logits = logits[mask]
                distances = distances[mask]
                labels = labels[mask]
                
            if calculate_scores:
                softmin = softmax(-distances)
                invScores = 1-softmin
                scores = distances*invScores
            else:
                if data_idx == 0:
                    scores = logits
                if data_idx == 1:
                    scores = distances

            X += scores.detach().cpu().tolist()
            # y += targets.cpu().tolist()
            y += labels.cpu().tolist()

    X = np.asarray(X)
    y = np.asarray(y)

    prediction = [j for i in prediction for j in i]
    feature = temp[1:, ::, ::, ::]
    feature = feature.view(len(feature), -1)
    feature = feature.cpu()
    

    return X, y, prediction, feature

def auroc(X, y):
    X = np.min(X, 1)

    temp = [0 for i in range(len(y))]
    for i in range(len(y)):
        if y[i]>=0:
            temp[i]=1

    label = temp
    
    return X, label

method_list = {}
class regmethod:

    def __init__(self,name) -> None:
        self.name = name
    
    def __call__(self,func, *args, **kwds):
        global method_list
        method_list[self.name] = func
        print("Registering",self.name)
        return func
    