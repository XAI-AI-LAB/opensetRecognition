from methods.augtools import HighlyCustomizableAugment, RandAugmentMC
import methods.util as util
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import random
from sklearn import metrics
from methods.util import AverageMeter
import time
from torchvision.transforms import transforms
from methods.resnet import ResNet
from torchvision import models as torchvision_models
from methods import PAN
from configs.pan import pan_r18_fpem_v1

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class GramRecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.gram_feats = []
        self.collecting = False
    
    def begin_collect(self,):
        self.gram_feats.clear()
        self.collecting = True
        # print("begin collect")

    def record(self,ft):
        if self.collecting:
            self.gram_feats.append(ft)
            # print("record")
    
    def obtain_gram_feats(self,):
        tmp = self.gram_feats
        self.collecting = False
        self.gram_feats = []
        # print("record")
        return tmp


class PretrainedResNet(nn.Module):

    def __init__(self,rawname, pretrain_path=None, args=None) -> None:
        super().__init__()
        self.args = args
        if pretrain_path == 'default':
            self.model = torchvision_models.__dict__[rawname](pretrained = True)
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
        else:
            self.model = torchvision_models.__dict__[rawname]()
            self.output_dim = self.model.fc.weight.shape[1]
            self.model.fc = nn.Identity()
            if pretrain_path is not None:
                sd = torch.load(pretrain_path)
                self.model.load_state_dict(sd,strict = True)

    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        if self.args.use_neck:
            f = []
            x = self.model.layer1(x)
            f.append(x)
            x = self.model.layer2(x)
            f.append(x)
            x = self.model.layer3(x)
            f.append(x)
            x = self.model.layer4(x)
            f.append(x)
            return tuple(f)
        else:
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            return x


class Backbone(nn.Module):

    def __init__(self,config, inchan, args):
        super().__init__()

        if config['backbone'] == 'resnet18':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], inchan=inchan, args=args)
        elif config['backbone'] == 'resnet18a':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], resfirststride=2, inchan=inchan, args=args)
        elif config['backbone'] == 'resnet18b':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], resfirststride=2, inchan=inchan, args=args)
        elif config['backbone'] == 'resnet34':
            self.backbone = ResNet(output_dim=config['category_model']['projection_dim'], num_block=[3,4,6,3], inchan=inchan, args=args)
        elif config['backbone'] in ['prt_r18','prt_r34','prt_r50']:
            self.backbone = PretrainedResNet(
                {'prt_r18':'resnet18','prt_r34':'resnet34','prt_r50':'resnet50'}[config['backbone']], args)
        elif config['backbone'] in ['prt_pytorchr18','prt_pytorchr34','prt_pytorchr50']:
            name,path = {
                'prt_pytorchr18':('resnet18','default'),
                'prt_pytorchr34':('resnet34','default'),
                'prt_pytorchr50':('resnet50','default')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path, args)
        elif config['backbone'] in ['prt_dinor18','prt_dinor34','prt_dinor50']:
            name,path = {
                'prt_dinor50':('resnet50','./model_weights/dino_resnet50_pretrain.pth')
            }[config['backbone']]
            self.backbone = PretrainedResNet(name, path, args)
        else:
            bkb = config['backbone']
            raise Exception(f'Backbone \"{bkb}\" is not defined.')

        # types : ae_softmax_avg , ae_avg_softmax , avg_ae_softmax
        self.output_dim = self.backbone.output_dim
        # self.classifier = CRFClassifier(self.backbone.output_dim,numclss,config)
        
    def forward(self,x):
        x = self.backbone(x)
        # latent , global prob , logits
        return x


class CACClassifier(nn.Module):
    def __init__(self, inchannels, num_class, config, anchor):
        super(CACClassifier, self).__init__()        
        self.num_classes = num_class
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        self.class_aes = []
        for i in range(self.num_classes):
            ae = AutoEncoder(inchannels, ae_hidden, ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        self.useL1 = config['error_measure'] == 'L1'

        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.anchors = nn.Parameter(anchor.double(), requires_grad=False).cuda()

    def set_anchors(self, means):
            self.anchors = nn.Parameter(means.double(), requires_grad=False)
            self.cuda()

    
    def distance_classifier(self, x):
        ''' Calculates euclidean distance from x to each class anchor
            Returns n x m array of distance from input of batch_size n to anchors of size m
        '''
        
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()	# 128 x 6 -> 128 x 6 x 6	      
        anchors = self.anchors.unsqueeze(0).expand(n, m, d) 	# 6 x 6 -> 128 x 6 x 6
        dists = torch.norm(x-anchors, 2, 2)	# 128 x 6

        return dists

    def ae_error(self,rc,x):
        if self.useL1:
            # return torch.sum(torch.abs(rc-x) * self.reduction, dim=1, keepdim=True)
            return torch.norm(rc - x,p = 1,dim = 1,keepdim=True) * self.reduction
        else:
            return torch.norm(rc - x,p = 2,dim = 1,keepdim=True) ** 2 * self.reduction

    clip_len = 100
    
    def forward(self,x):
        cls_ers = []
        for i in range(len(self.class_aes)):
            rc,lt = self.class_aes[i](x)
            cls_er = self.ae_error(rc,x)
            if CACClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er,-CACClassifier.clip_len,CACClassifier.clip_len)
            cls_ers.append(cls_er)
        logits = torch.cat(cls_ers,dim=1)
        probs = self.avg_pool(logits).view(logits.shape[0], -1)
        
        outDistance = self.distance_classifier(probs)
        
        return (probs, outDistance)
        
        
class LinearClassifier(nn.Module):

    def __init__(self,inchannels,num_class, config ,anchor):
        super().__init__()
        self.gamma = config['gamma']
        self.cls = nn.Conv2d(inchannels, num_class , 1,padding= 0, bias=False)
    
    def forward(self,x):
        x = self.cls(x)
        return x * self.gamma


def sim_conv_layer(input_channel,output_channel,kernel_size=1,padding =0,use_activation = True):
    if use_activation :
        res = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size,padding= padding, bias=False),
                nn.Tanh())
    else:
        res = nn.Conv2d(input_channel, output_channel, kernel_size,padding= padding, bias=False)
    return res


class AutoEncoder(nn.Module):

    def __init__(self,inchannel,hidden_layers,latent_chan):
        super().__init__()
        layer_block = sim_conv_layer
        self.latent_size = latent_chan
        if latent_chan > 0:
            self.encode_convs = []
            self.decode_convs = []
            for i in range(len(hidden_layers)):
                h = hidden_layers[i]
                ecv = layer_block(inchannel,h,)
                dcv = layer_block(h,inchannel,use_activation = i != 0)
                inchannel = h
                self.encode_convs.append(ecv)
                self.decode_convs.append(dcv)
            self.encode_convs = nn.ModuleList(self.encode_convs)
            self.decode_convs.reverse()
            self.decode_convs = nn.ModuleList(self.decode_convs)
            self.latent_conv = layer_block(inchannel,latent_chan)
            self.latent_deconv = layer_block(latent_chan,inchannel,use_activation = (len(hidden_layers) > 0))
        else:
            self.center = nn.Parameter(torch.rand([inchannel,1,1]),True)
    
    def forward(self,x):
        if self.latent_size > 0:
            output = x
            for cv in self.encode_convs:
                output = cv(output)
            latent = self.latent_conv(output)
            output = self.latent_deconv(latent)
            for cv in self.decode_convs:
                output = cv(output)
            return output,latent
        else:
            return self.center,self.center


class CSSRClassifier(nn.Module):

    def __init__(self,inchannels,num_class, config, anchor):
        super().__init__()
        ae_hidden = config['ae_hidden']
        ae_latent = config['ae_latent']
        self.class_aes = []
        for i in range(num_class):
            ae = AutoEncoder(inchannels,ae_hidden,ae_latent)
            self.class_aes.append(ae)
        self.class_aes = nn.ModuleList(self.class_aes)
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.useL1 = config['error_measure'] == 'L1'

        self.reduction = -1 if config['model'] == 'pcssr' else 1
        self.reduction *= config['gamma']

    
    def ae_error(self,rc,x):
        if self.useL1:
            return torch.norm(rc - x,p = 1,dim = 1,keepdim=True) * self.reduction
        else:
            return torch.norm(rc - x,p = 2,dim = 1,keepdim=True) ** 2 * self.reduction
        
    def mahalanobis(self, u, v, cov):
        delta = u - v
        m = torch.dot(delta, torch.matmul(torch.inverse(cov), delta))
        return torch.sqrt(m)

    clip_len = 100

    def forward(self,x):
        cls_ers = []
        for i in range(len(self.class_aes)):
            rc,lt = self.class_aes[i](x)
            cls_er = self.ae_error(rc,x)

            if CSSRClassifier.clip_len > 0:
                cls_er = torch.clamp(cls_er,-CSSRClassifier.clip_len,CSSRClassifier.clip_len)
            cls_ers.append(cls_er)
            
        logits = torch.cat(cls_ers,dim=1)
        return logits

def G_p(ob, p):
    temp = ob.detach()
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))#
    temp = temp.reshape([temp.shape[0],-1])#.sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)
    
    return temp


def G_p_pro(ob, p = 8):
    temp = ob.detach()
    temp = temp**p
    temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))#
    # temp = temp.reshape([temp.shape[0],-1])#.sum(dim=2) 
    temp = (temp.sign()*torch.abs(temp)**(1/p))#.reshape(temp.shape[0],ob.shape[1],ob.shape[1])
    
    return temp

def G_p_inf(ob,p = 1):
    temp = ob.detach()
    temp = temp**p
    # print(temp.shape)
    temp = temp.reshape([temp.shape[0],temp.shape[1],-1]).transpose(dim0=2,dim1=1).reshape([-1,temp.shape[1],1])
    # print(temp.shape)
    temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1))))#
    temp = (temp.sign()*torch.abs(temp)**(1/p))
    # print(temp.shape)
    return temp.reshape(ob.shape[0],ob.shape[2],ob.shape[3],ob.shape[1],ob.shape[1])
    
# import methods.pooling.MPNConv as MPN

class BackboneAndClassifier(nn.Module):

    def __init__(self,num_classes,config, args, anchor=None):
        super().__init__()
        self.args = args
        clsblock = {'linear':LinearClassifier,'pcssr':CSSRClassifier,'rcssr' : CSSRClassifier, 'cac': CACClassifier}
        self.backbone = Backbone(config,3, self.args)
        cfg_neck = pan_r18_fpem_v1.model['neck']
        output_featuremap = pan_r18_fpem_v1.model['neck']['out_feature_map']
        self.neck = PAN(self.backbone, cfg_neck, output_featuremap)
        cat_config = config['category_model']
        self.cat_cls = clsblock[cat_config['model']](self.backbone.output_dim,num_classes,cat_config, anchor)

    def forward(self,x,feature_only = False):
        if self.args.use_neck:
            x = self.neck(x)
            if feature_only:
                return x
        else:
            x = self.backbone(x)
            if feature_only:
                return x
        return x, self.cat_cls(x)

    
class CSSRModel(nn.Module):

    def __init__(self,num_classes,config,crt, args, anchor=None):
        super().__init__()
        self.crt = crt
        self.args = args
        
        # ------ New Arch
        if config['category_model']['model'] == 'cac':
            self.backbone_cs = BackboneAndClassifier(num_classes,config, self.args, anchor)
        else:
            self.backbone_cs = BackboneAndClassifier(num_classes, config, self.args)

        self.config = config
        self.mins = {i : [] for i in range(num_classes)}
        self.maxs = {i : [] for i in range(num_classes)}
        self.num_classes = num_classes

        self.avg_feature = [[0,0] for i in range(num_classes)]
        self.avg_gram = [[[0,0] for i in range(num_classes)] for i in self.powers]
        self.enable_gram = config['enable_gram']
    
    def update_minmax(self,feat_list,power = [],ypred = None):
        # feat_list = self.gram_feature_list(batch)
        for pr in range(self.num_classes):
            cond = ypred == pr
            if not cond.any():
                continue
            for L,feat_L in enumerate(feat_list):
                if L==len(self.mins[pr]):
                    self.mins[pr].append([None]*len(power))
                    self.maxs[pr].append([None]*len(power))
                
                for p,P in enumerate(power):
                    g_p = G_p(feat_L[cond],P)
                    
                    current_min = g_p.min(dim=0,keepdim=True)[0]
                    current_max = g_p.max(dim=0,keepdim=True)[0]
                    
                    if self.mins[pr][L][p] is None:
                        self.mins[pr][L][p] = current_min
                        self.maxs[pr][L][p] = current_max
                    else:
                        self.mins[pr][L][p] = torch.min(current_min,self.mins[pr][L][p])
                        self.maxs[pr][L][p] = torch.max(current_max,self.maxs[pr][L][p])


    def get_deviations(self,feat_list,power,ypred):
        batch_deviations = None
        for pr in range(self.num_classes):
            mins,maxs = self.mins[pr],self.maxs[pr]
            cls_batch_deviations = []
            cond = ypred==pr
            if not cond.any():
                continue
            for L,feat_L in enumerate(feat_list):
                dev = 0
                for p,P in enumerate(power):
                    g_p = G_p(feat_L[cond],P)
                    # print(L,len(mins))
                    # print(p,len(mins[L]))
                    dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                cls_batch_deviations.append(dev.cpu().detach().numpy())
            cls_batch_deviations = np.concatenate(cls_batch_deviations,axis=1)
            if batch_deviations is None:
                batch_deviations = np.zeros([ypred.shape[0],cls_batch_deviations.shape[1]])
            batch_deviations[cond] = cls_batch_deviations
        return batch_deviations
    
    powers = [8]

    def cal_feature_prototype(self,feat,ypred):
        feat = torch.abs(feat)
        for pr in range(self.num_classes):
            cond = ypred==pr
            if not cond.any():
                continue
            csfeat = feat[cond]
            cf = csfeat.mean(dim = [0,2,3]) #.cpu().numpy()
            ct = cond.sum()
            ft = self.avg_feature[pr]
            self.avg_feature[pr] = [ft[0] + ct, (ft[1] * ft[0] + cf * ct)/(ft[0] + ct)]
            if self.enable_gram:
                for p in range(len(self.powers)):
                    gram = G_p_pro(csfeat,self.powers[p]).mean(dim = 0)
                    gm = self.avg_gram[p][pr]
                    self.avg_gram[p][pr] = [gm[0] + ct, (gm[1] * gm[0] + gram * ct)/(gm[0] + ct)]

    def obtain_usable_feature_prototype(self):
        if isinstance(self.avg_feature,list):
            clsft_lost = []
            exm = None
            for x in self.avg_feature:
                if x[0] > 0:
                    clsft_lost.append(x[1])
                    exm = x[1]
                else:
                    clsft_lost.append(None)
            clsft = torch.stack([torch.zeros_like(exm) if x is None else x for x in clsft_lost])
            print(clsft.sum(dim=0).shape)
            # print(np.isnan(clsft.cpu().numpy()).any())
            clsft /= clsft.sum(dim = 0) #**2
            # clsft /= clsft.sum(dim = 1,keepdim = True)
            self.avg_feature = clsft.reshape([clsft.shape[0],1,clsft.shape[1],1,1])
            if self.enable_gram:
                for i in range(len(self.powers)):
                    self.avg_gram[i] = torch.stack([x[1] if x[0] > 0 else torch.zeros([exm.shape[0],exm.shape[0]]).cuda() for x in self.avg_gram[i]])
            # self.avg_gram /= self.avg_gram.sum(dim = 0)
        return self.avg_feature,self.avg_gram

    def get_feature_prototype_deviation(self,feat,ypred):
        # feat = torch.abs(feat)
        avg_feature,_ = self.obtain_usable_feature_prototype()
        scores = np.zeros([feat.shape[0],feat.shape[2],feat.shape[3]])  # 128 * 4 * 4
        for pr in range(self.num_classes):
            cond = ypred==pr
            if not cond.any():
                continue
            scores[cond] = (avg_feature[pr] * feat[cond]).mean(axis = 1).cpu().numpy()
            # print(np.isnan(avg_feature[pr].cpu().numpy()).any())

        return scores
    
    def get_feature_gram_deviation(self,feat,ypred):
        _,avg_gram = self.obtain_usable_feature_prototype()
        scores = np.zeros([feat.shape[0],feat.shape[2],feat.shape[3]])
        for pr in range(self.num_classes):
            cond = ypred==pr
            if not cond.any():
                continue
            res = 0
            for i in range(len(self.powers)):
                gm = G_p_pro(feat[cond],p=self.powers[i])
                # scores[cond] = (gm / gm.mean(dim = [3,4],keepdim = True) * avg_gram[pr]).sum(dim = [3,4]).cpu().numpy()
                res += (gm * avg_gram[i][pr]).sum(dim = [1,2],keepdim = True).cpu().numpy()
            scores[cond] = res
        return scores
    
    def pred_by_feature_gram(self,feat):
        _,avg_gram = self.obtain_usable_feature_prototype()
        scores = np.zeros([self.num_classes, feat.shape[0]])
        gm = G_p_pro(feat)
        for pr in range(self.num_classes):
            # scores[cond] = (gm / gm.mean(dim = [3,4],keepdim = True) * avg_gram[pr]).sum(dim = [3,4]).cpu().numpy()
            scores[pr] = (gm * avg_gram[pr]).sum(dim = [1,2]).cpu().numpy()
        return scores.argmax(axis = 0)

    def forward(self,x,ycls = None,reqpredauc = False,prepareTest = False,reqfeature = False):
        
        # ----- New Arch
        x = self.backbone_cs(x,feature_only = reqfeature)
        if reqfeature:
           return x
        x,xcls_raw = x

        def pred_score(xcls):
            score_reduce = lambda x : x.reshape([x.shape[0],-1]).mean(axis = 1)
            x_detach = x.detach()
            probs = self.crt(xcls,prob = True).cpu().numpy()
            pred = probs.argmax(axis = 1)
            max_prob = probs.max(axis = 1)

            cls_scores = xcls.cpu().numpy()[[i for i in range(pred.shape[0])],pred]
            rep_scores = torch.abs(x_detach).mean(dim = 1).cpu().numpy()
            if not self.training and not prepareTest and (not isinstance(self.avg_feature,list) or  self.avg_feature[0][0] != 0):
                rep_cspt = self.get_feature_prototype_deviation(x_detach,pred)
                if self.enable_gram:
                    rep_gram = self.get_feature_gram_deviation(x_detach,pred)
                else:
                    rep_gram = np.zeros_like(cls_scores)
            else:
                rep_cspt = np.zeros_like(cls_scores)
                rep_gram = np.zeros_like(cls_scores)
            R = [cls_scores,rep_scores,rep_cspt,rep_gram,max_prob]

            scores = np.stack([score_reduce(eval(self.config['score'])),score_reduce(rep_cspt),score_reduce(rep_gram)],axis = 1)
            return pred,scores

        if self.training:
            if self.config['category_model']['model'] == 'cac':
                xcls, anchor = self.crt(xcls_raw, ycls)
                # probs = xcls_raw[0]
                # pred = probs.argmax(axis=1)
                _, pred = xcls_raw[1].min(1)
                return xcls, pred, anchor
            else:
                xcls = self.crt(xcls_raw, y=ycls)
            if reqpredauc:
                pred,score = pred_score(xcls_raw.detach())
                return xcls,pred,score

        else:
            if self.config['category_model']['model'] == 'cac':
                # xcls, anchor = self.crt(xcls_raw, ycls)
                # probs = xcls_raw[0]
                _, pred = xcls_raw[1].min(1)
                # pred = probs.argmax(axis=1)
                x_detach = x.detach()
                return xcls_raw, pred, x_detach
            else:    
                xcls = xcls_raw
                # xrot = self.rot_cls(x)
                if reqpredauc:
                    pred,score = pred_score(xcls)
                    deviations = None
                    # powers = range(1,10)
                    if prepareTest:
                        if not isinstance(self.avg_feature,list):
                            self.avg_feature = [[0,0] for i in range(self.num_classes)]
                            self.avg_gram = [[[0,0] for i in range(self.num_classes)] for i in self.powers]
                        # hdfts = self.backbone.backbone.obtain_gram_feats()
                        # self.update_minmax(hdfts + [x] + clslatents,powers,pred)
                        self.cal_feature_prototype(x,pred)
                    # else:
                        # deviations = self.get_deviations(self.backbone.backbone.obtain_gram_feats() + [x]+ clslatents,powers,pred)
                    return x, pred,score,deviations
        
        
        return xcls



class CSSRCriterion(nn.Module):
    # prob(g) -> CAC -> CAC loss(output)

    def get_onehot_label(self,y,clsnum):
        y = torch.reshape(y,[-1,1]).long()
        return torch.zeros(y.shape[0], clsnum).cuda().scatter_(1, y, 1)
        
    def CACLoss(self, distances, gt, clsnum):
        '''Returns CAC loss, as well as the Anchor and Tuplet loss components separately for visualisation.'''
        lbda = 0.1
        true = torch.gather(distances, 1, gt.view(-1, 1)).view(-1)
        non_gt = torch.Tensor([[i for i in range(clsnum) if gt[x] != i] for x in range(len(distances))]).long().cuda()
        others = torch.gather(distances, 1, non_gt)
	
        anchor = torch.mean(true)

        tuplet = torch.exp(-others+true.unsqueeze(1))
        tuplet = torch.mean(torch.log(1+torch.sum(tuplet, dim = 1)))

        total = lbda*anchor + tuplet

        return total, anchor, tuplet

    def __init__(self,avg_order, clsnum, totalnum, cls_list, enable_sigma=True, cac=False):
        super().__init__()
        self.avg_order = {"avg_softmax":1,"softmax_avg":2}[avg_order]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enable_sigma = enable_sigma
        self.clsnum = clsnum
        self.cac = cac

    def forward(self, x, y=None, prob=False, pred=False):
        if self.cac:
            x1, x2 = x
            total, anchor, tuplet = self.CACLoss(x2, y, self.clsnum)
            return total, anchor
        else:
            x = x
            if self.avg_order == 1:
                g = self.avg_pool(x).view(x.shape[0],-1)
                g = torch.softmax(g,dim=1)
            elif self.avg_order == 2:
                g = torch.softmax(x,dim=1)
                g = self.avg_pool(g).view(x.size(0), -1)
            if prob: return g
            if pred: return torch.argmax(g,dim = 1)

            loss = -torch.sum(self.get_onehot_label(y,g.shape[1]) * torch.log(g),dim=1).mean()
            return loss


def manual_contrast(x):
    s = random.uniform(0.1,2)
    return x * s


class WrapDataset(data.Dataset):

    def __init__(self,labeled_ds,config,inchan_num = 3) -> None:
        super().__init__()
        self.labeled_ds = labeled_ds

        __mean = [0.5,0.5,0.5][:inchan_num]
        __std = [0.25,0.25,0.25][:inchan_num]
            
        trans = [transforms.RandomHorizontalFlip()]
        if config['cust_aug_crop_withresize']:
            trans.append(transforms.RandomResizedCrop(size = util.img_size,scale = (0.25,1)))
        elif util.img_size > 200:
            trans += [transforms.Resize(256),transforms.RandomResizedCrop(util.img_size)]
        else:
            trans.append(transforms.RandomCrop(size=util.img_size,
                                    padding=int(util.img_size*0.125),
                                    padding_mode='reflect'))
        if config['strong_option'] == 'RA':
            trans.append(RandAugmentMC(n=2, m=10))
        elif config['strong_option'] == 'CUST':
            trans.append(HighlyCustomizableAugment(2,10,-1,labeled_ds,config))
        elif config['strong_option'] == 'NONE':
            pass
        else:
            raise NotImplementedError()
        trans += [transforms.ToTensor(),
                  transforms.Normalize(mean=__mean, std=__std)]
        
        if config['manual_contrast']:
            trans.append(manual_contrast)
        strong = transforms.Compose(trans)

        if util.img_size > 200:
            self.simple = [transforms.RandomResizedCrop(util.img_size)]
        else:
            self.simple = [transforms.RandomCrop(size=util.img_size,
                                            padding=int(util.img_size*0.125),
                                            padding_mode='reflect')]
        self.simple = transforms.Compose(([transforms.RandomHorizontalFlip()]) + self.simple + [
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=__mean, std=__std)] + ([manual_contrast] if config['manual_contrast'] else []))

        self.test_normalize = transforms.Compose([
                                    transforms.CenterCrop(util.img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=__mean, std=__std)])

        td = {'strong' : strong, 'simple' : self.simple}
        self.aug = td[config['cat_augmentation']]
        self.test_mode = False

    def __len__(self) -> int:
        return len(self.labeled_ds)
    
    def __getitem__(self, index: int) :
        img,lb,_ = self.labeled_ds[index]
        if self.test_mode:
            img = self.test_normalize(img)
        else:
            img = self.aug(img)
        return img,lb,index


@util.regmethod('cssr')
class CSSRMethod:

    def get_cfg(self,key,default):
        return self.config[key] if key in self.config else default
    
    def __init__(self, config, clssnum,totalnum, cls_list,train_loader, args, saving_path, datasets) -> None:
        self.config = config
        self.cat_config = config['category_model']
        self.epoch = 0
        self.lr = config['learn_rate']
        self.batch_size = config['batch_size']

        self.datasets = datasets
        self.saving_path = saving_path
        self.args = args
        self.clsnum = clssnum
        self.totalnum = totalnum
        self.cls_list = cls_list
        self.train_loader = train_loader
        self.anchors = torch.Tensor(torch.zeros(clssnum, clssnum))
        
        self.wrap_ds = WrapDataset(train_loader.dataset,self.config,inchan_num=3,)
        self.wrap_loader = data.DataLoader(self.wrap_ds,
            batch_size=self.config['batch_size'], shuffle=True,pin_memory=True, num_workers=6)
        self.lr_schedule = util.get_scheduler(self.config,self.wrap_loader)

        if self.cat_config['model'] == 'cac':
            self.crt = CSSRCriterion(config['arch_type'], self.clsnum, self.totalnum, cls_list,False, True)
            # initialising with anchors
            self.anchors = torch.diag(torch.Tensor([10 for i in range(self.clsnum)]))
            self.model = CSSRModel(self.clsnum, config, self.crt, self.args, self.anchors).cuda()
            
            if self.args.head_only:
                self.model.backbone_cs.requires_grad = False
                self.model.backbone_cs.neck.requires_grad = False
                print(f'[Info] freezed {self.cat_config["model"]} backbone({self.config["backbone"]})')
        else:
            self.crt = CSSRCriterion(config['arch_type'], self.clsnum, self.totalnum, cls_list,False, False)
            self.model = CSSRModel(self.clsnum, config, self.crt, self.args).cuda()

        self.modelopt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        self.prepared = -999

    def train_epoch(self):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        train_acc = AverageMeter()

        running_loss = AverageMeter()
        if self.cat_config['model'] == 'cac':
            train_loss = 0
            correctDist = 0
            total = 0
            
        self.model.train()

        endtime = time.time()
        for i, data in enumerate(tqdm.tqdm(self.wrap_loader)):
            data_time.update(time.time() - endtime)

            self.lr = self.lr_schedule.get_lr(self.epoch,i,self.lr)
            util.set_lr([self.modelopt],self.lr)
            sx, lb = data[0].cuda(),data[1].cuda()
            
            if self.cat_config['model'] == 'cac':
                cacLoss,pred, anchor = self.model(sx, lb, reqpredauc=True)
                
                self.modelopt.zero_grad()
                cacLoss.backward()
                self.modelopt.step()
                
                nplb = data[1].numpy()
                pred = pred.detach().cpu().numpy()
                
                train_loss += cacLoss.item()
                # total += nplb.shape[0]
                # correctDist += ((pred == nplb).sum())
                train_acc.update((pred == nplb).sum() / pred.shape[0], pred.shape[0])
                
                running_loss.update(cacLoss.item())
                batch_time.update(time.time() - endtime)
                endtime = time.time()
                
            else:
                loss,pred,scores = self.model(sx, lb, reqpredauc=True)
                
                self.modelopt.zero_grad()
                loss.backward()
                self.modelopt.step()
                
                
                nplb = data[1].numpy()
                train_acc.update((pred == nplb).sum() / pred.shape[0],pred.shape[0])
                running_loss.update(loss.item())
                batch_time.update(time.time() - endtime)
                endtime = time.time()
                
        self.epoch += 1
        if self.cat_config['model'] == 'cac':
            training_res = \
                {"Loss" : running_loss.avg,
                # {"Loss" : train_loss/(i+1),
                # "TrainAcc" : 100.*correctDist/total,
                "TrainAcc" : train_acc.avg,
                "Learn Rate" : self.lr,
                "DataTime" : data_time.avg,
                "BatchTime" : batch_time.avg}
        else:
            training_res = \
                    {"Loss" : running_loss.avg,
                    "TrainAcc" : train_acc.avg,
                    "Learn Rate" : self.lr,
                    "DataTime" : data_time.avg,
                    "BatchTime" : batch_time.avg}

        return training_res

    
    def known_prediction_test(self,test_loader):
        self.model.eval()
        pred,scores,_,_ = self.scoring(test_loader)
        return pred

    def scoring(self, loader, prepare=False):
        gts = []
        deviations = []
        scores = []
        prediction = []
        if self.args.use_neck:
            if self.datasets == "tinyimagenet":
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
            if self.datasets == "tinyimagenet":
                temp = torch.zeros(1,512,8,8).cuda()
            else:
                temp = torch.zeros(1,512,4,4).cuda()
        
        with torch.no_grad():
            for d in tqdm.tqdm(loader):
                x1 = d[0].cuda(non_blocking = True)
                gt = d[1].numpy()
                feat,pred,scr,dev = self.model(x1,reqpredauc = True,prepareTest = prepare)

                prediction.append(pred)
                scores.append(scr)
                gts.append(gt)
                temp = torch.cat([temp, feat], dim=0)

        self.feature = temp[1:, ::, ::, ::]
        prediction = np.concatenate(prediction)
        scores = np.concatenate(scores)
        gts = np.concatenate(gts)

        return prediction,scores,deviations,gts

    def close_labels(self,test_loader):
        labels = test_loader.dataset.labels
        test_labels = np.array(labels,np.int)
        close_samples = test_labels >= 0
        return close_samples

    def visualize_feature(self, close_samples, features, pred, scores, gts, config, args):
            open_pred = pred.copy()
            open_pred = np.array(open_pred)
            thresh = -9999999
            if thresh < -99999:
                fpr, tpr, thresholds  =  metrics.roc_curve(close_samples, scores) 
                thresh = thresholds[np.abs(np.array(tpr) - 0.95).argmin()]
            open_pred[scores <= thresh] = -1
            util.visualize(features, gts, pred, open_pred, config, args, self.saving_path)

    def knownpred_unknwonscore_cssr_test(self,test_loader, test=False):
        self.model.eval()
        if self.prepared != self.epoch:
            self.wrap_ds.test_mode = True
            tpred,tscores,_,_ = self.scoring(self.wrap_loader, True)

            self.wrap_ds.test_mode = False
            self.prepared = self.epoch
        pred,scores,devs,gts = self.scoring(test_loader)
        feature = self.feature

        if self.config['integrate_score'] != "S[0]":
            tpred,tscores,_,_ = self.scoring(self.wrap_loader, False)
            mean,std = tscores.mean(axis = 0),tscores.std(axis = 0)
            scores = (scores - mean)/(std + 1e-8)
        S = scores.T
        scores = eval(self.config['integrate_score'])
        if test:
            if self.config['manifold']:
                close_samples = self.close_labels(test_loader)
                self.visualize_feature(close_samples, feature, pred, scores, gts, self.config, self.args)

        return scores,-9999999,pred

    def knownpred_unknwonscore_cac_test(self,test_loader, test=False):
        self.model.eval()
        #find mean anchors for each class
        # anchor_means = util.find_anchor_means(self.model, test_loader, self.clsnum, only_correct = True)
        self.anchors = torch.diag(torch.Tensor([10 for i in range(self.clsnum)]))
        output_featuremap = pan_r18_fpem_v1.model['neck']['out_feature_map']
        scores, gts, pred, feature = util.gather_outputs(self.model, test_loader, 
                                                         output_feature=output_featuremap, 
                                                         data_idx=1, 
                                                         calculate_scores=True, 
                                                         args=self.args,
                                                         datasets=self.datasets)

        scores, onehot_label = util.auroc(scores, gts)

        self.feature = feature
        if test:
            if self.config['manifold']:
                close_samples = self.close_labels(test_loader)
                self.visualize_feature(close_samples, self.feature, pred, scores, gts, self.config, self.args)

        return scores, -9999999, pred


    def save_model(self,path):
        save_dict = {
            'model' : self.model.state_dict(),
            'config': self.config,
            'optimzer' : self.modelopt.state_dict(),
            'epoch' : self.epoch,
        }
        torch.save(save_dict,path)

    def load_model(self,path):
        if self.args.transfer_learning:
            save_dict = torch.load(path)
            self.model.load_state_dict(save_dict['model'], strict=False)
        else:
            save_dict = torch.load(path)
            self.model.load_state_dict(save_dict['model'], strict=False)
            if 'optimzer' in save_dict and self.modelopt is not None:
                self.modelopt.load_state_dict(save_dict['optimzer'])
            self.epoch = save_dict['epoch']
