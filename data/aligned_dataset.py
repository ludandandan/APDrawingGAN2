import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import cv2
import csv

def getfeats(featpath):
	trans_points = np.empty([5,2],dtype=np.int64) 
	with open(featpath, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		for ind,row in enumerate(reader):
			trans_points[ind,:] = row
	return trans_points

def tocv2(ts):
    img = (ts.numpy()/2+0.5)*255
    img = img.astype('uint8')
    img = np.transpose(img,(1,2,0))
    img = img[:,:,::-1]#rgb->bgr
    return img

def dt(img):
    if(img.shape[2]==3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #convert to BW
    ret1,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret2,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    dt1 = cv2.distanceTransform(thresh1,cv2.DIST_L2,5)
    dt2 = cv2.distanceTransform(thresh2,cv2.DIST_L2,5)
    dt1 = dt1/dt1.max()#->[0,1]
    dt2 = dt2/dt2.max()
    return dt1, dt2

def getSoft(size,xb,yb,boundwidth=5.0):
    xarray = np.tile(np.arange(0,size[1]),(size[0],1)) # x坐标的数组网格，就是列，np.tile是复制数组的
    yarray = np.tile(np.arange(0,size[0]),(size[1],1)).transpose()# y坐标的数组网格，就是行
    cxdists = [] # 分别存储 x 和 y 方向上每个坐标点到给定坐标 (xb, yb) 的距离。
    cydists = []
    for i in range(len(xb)):
        xba = np.tile(xb[i],(size[1],1)).transpose()
        yba = np.tile(yb[i],(size[0],1))
        cxdists.append(np.abs(xarray-xba))
        cydists.append(np.abs(yarray-yba))
    xdist = np.minimum.reduce(cxdists) # 计算x，y方向上的最小距离
    ydist = np.minimum.reduce(cydists)
    manhdist = np.minimum.reduce([xdist,ydist]) # 计算曼哈顿距离，即x和y方向上最小距离的最小值
    im = (manhdist+1) / (boundwidth+1) * 1.0 # 对曼哈顿距离进行归一化和平滑，得到软边界图像
    im[im>=1.0] = 1.0
    return im

class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        imglist = 'datasets/apdrawing_list/%s/%s.txt' % (opt.phase, opt.dataroot)
        if os.path.exists(imglist):
            lines = open(imglist, 'r').read().splitlines()
            lines = sorted(lines)
            self.AB_paths = [line.split()[0] for line in lines]
            if len(lines[0].split()) == 2:
                self.B_paths = [line.split()[1] for line in lines]
        else:
            self.dir_AB = os.path.join(opt.dataroot, opt.phase)
            self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        if w/h == 2:
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = AB.crop((w2, 0, w, h)).resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        else: # if w/h != 2, need B_paths
            A = AB.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            B = Image.open(self.B_paths[index]).convert('RGB')
            B = B.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]#C,H,W
        B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        flipped = False
        if (not self.opt.no_flip) and random.random() < 0.5:
            flipped = True
            idx = [i for i in range(A.size(2) - 1, -1, -1)] # 把A宽度维度上的index整个翻转过来，比如A宽度是10，原来index是[0,1,2,...,9]，翻转之后变为[9,8,...,0]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)#选择在第3个维度上CHW，也就是W维度上进行反转，整个index反过来
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        
        item = {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

        if self.opt.use_local:
            regions = ['eyel','eyer','nose','mouth']
            basen = os.path.basename(AB_path)[:-4]+'.txt'
            if self.opt.region_enm in [0,1]: # 0是矩形区域，1是矩形区域内的紧凑掩膜，2是没有矩形区域的掩膜
                featdir = self.opt.lm_dir # dataset/landmark/，path to facial landmarks
                featpath = os.path.join(featdir,basen)
                feats = getfeats(featpath)
                if flipped:
                    for i in range(5):
                        feats[i,0] = self.opt.fineSize - feats[i,0] - 1 #左眼，右眼，鼻子，嘴巴水平翻转
                    tmp = [feats[0,0],feats[0,1]] # 左眼和右眼位置交换
                    feats[0,:] = [feats[1,0],feats[1,1]]
                    feats[1,:] = tmp
                mouth_x = int((feats[3,0]+feats[4,0])/2.0) #feats中第0列是宽度，x坐标，嘴巴x坐标的中心点
                mouth_y = int((feats[3,1]+feats[4,1])/2.0) #feats中第1列是高度，y坐标，嘴巴y坐标的中心点
                ratio = self.opt.fineSize / 256
                EYE_H = self.opt.EYE_H * ratio
                EYE_W = self.opt.EYE_W * ratio
                NOSE_H = self.opt.NOSE_H * ratio
                NOSE_W = self.opt.NOSE_W * ratio
                MOUTH_H = self.opt.MOUTH_H * ratio
                MOUTH_W = self.opt.MOUTH_W * ratio
                center = torch.IntTensor([[feats[0,0],feats[0,1]-4*ratio],[feats[1,0],feats[1,1]-4*ratio],[feats[2,0],feats[2,1]-NOSE_H/2+16*ratio],[mouth_x,mouth_y]])
                item['center'] = center
                rhs = [int(EYE_H),int(EYE_H),int(NOSE_H),int(MOUTH_H)]
                rws = [int(EYE_W),int(EYE_W),int(NOSE_W),int(MOUTH_W)]
                if self.opt.soft_border:
                    soft_border_mask4 = []
                    for i in range(4):
                        xb = [np.zeros(rhs[i]),np.ones(rhs[i])*(rws[i]-1)] #局部图像的x方向上两个边界
                        yb = [np.zeros(rws[i]),np.ones(rws[i])*(rhs[i]-1)] #局部图像的y方向上两个边界
                        soft_border_mask = getSoft([rhs[i],rws[i]],xb,yb)
                        soft_border_mask4.append(torch.Tensor(soft_border_mask).unsqueeze(0))
                        item['soft_'+regions[i]+'_mask'] = soft_border_mask4[i]
                for i in range(4):
                    item[regions[i]+'_A'] = A[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
                    item[regions[i]+'_B'] = B[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
                    if self.opt.soft_border:
                        item[regions[i]+'_A'] = item[regions[i]+'_A'] * soft_border_mask4[i].repeat(int(input_nc/output_nc),1,1)# A是3通道的，腌膜是单通道的，所以需要将腌膜repeat上3倍
                        item[regions[i]+'_B'] = item[regions[i]+'_B'] * soft_border_mask4[i]
            if self.opt.compactmask: # use compact mask as input and apply to loss
                cmasks0 = []
                cmasks = []
                for i in range(4):
                    if flipped and i in [0,1]: # 如果此次翻转，并且是左眼或右眼，那么对于Mask，左眼取右眼的mask，右眼取左眼的mask
                        cmaskpath = os.path.join(self.opt.cmask_dir,regions[1-i],basen[:-4]+'.png') 
                    else: # 否则的话到dataset/mask路径下取各自的局部mask
                        cmaskpath = os.path.join(self.opt.cmask_dir,regions[i],basen[:-4]+'.png')
                    im_cmask = Image.open(cmaskpath) # 获取到的mask是Image类型的
                    cmask0 = transforms.ToTensor()(im_cmask) #mask转化为tensor类型的，shape是3x512x512
                    if flipped:
                        cmask0 = cmask0.index_select(2, idx) # 如果此次翻转，就水平翻转
                    if output_nc == 1 and cmask0.shape[0] == 3:
                        tmp = cmask0[0, ...] * 0.299 + cmask0[1, ...] * 0.587 + cmask0[2, ...] * 0.114 #将输入张量 cmask0 的三个通道按照灰度图像的公式进行加权平均，其中权重分别为 0.299、0.587 和 0.114
                        cmask0 = tmp.unsqueeze(0) #然后在axis=0上插入一个新的维度，shape变为1x512x512
                    cmask0 = (cmask0 >= 0.5).float() # 二值化处理，>=0.5的设置为1，否则设置为0
                    cmasks0.append(cmask0) #将左眼、右眼、鼻子、嘴巴mask都存到masks0中，这里面存放的是1x512x512的
                    cmask = cmask0.clone()
                    if self.opt.region_enm in [0,1]:
                        cmask = cmask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)]
                    elif self.opt.region_enm in [2]: # need to multiply cmask
                        item[regions[i]+'_A'] = (A/2+0.5) * cmask * 2 - 1
                        item[regions[i]+'_B'] = (B/2+0.5) * cmask * 2 - 1
                    cmasks.append(cmask) #裁剪之后的mask存放在cmasks里面
                item['cmaskel'] = cmasks[0] #裁剪之后的mask存放在字典item里面
                item['cmasker'] = cmasks[1]
                item['cmask'] = cmasks[2]
                item['cmaskmo'] = cmasks[3]
            if self.opt.hair_local: # add hair part
                mask = torch.ones(B.shape) # 1x512x512
                if self.opt.region_enm == 0: # 矩形区域方式
                    for i in range(4): # mask眼睛鼻子嘴巴位置设置为0，其他位置还是1
                        mask[:,int(center[i,1]-rhs[i]/2):int(center[i,1]+rhs[i]/2),int(center[i,0]-rws[i]/2):int(center[i,0]+rws[i]/2)] = 0
                    if self.opt.soft_border: # 如果使用软边界
                        imgsize = self.opt.fineSize
                        maskn = mask[0].numpy()# shape是512x512，里面的数值眼睛鼻子嘴是0，其他全是1
                        masks = [np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize]),np.ones([imgsize,imgsize])]
                        masks[0][1:] = maskn[:-1] #将列表 maskn 中除最后一个元素之外的部分赋值给列表 masks 中的第一个元素（除第一个元素之外的部分
                        masks[1][:-1] = maskn[1:] #将列表 maskn 中除第一个元素之外的部分赋值给列表 masks 中的第二个元素（除最后一个元素之外的部分）
                        masks[2][:,1:] = maskn[:,:-1]#将列表 maskn 中的每一行的第一列开始到倒数第二列的部分赋值给列表 masks 中的第三个元素（每一行的第二列开始到最后一列的部分
                        masks[3][:,:-1] = maskn[:,1:] #将列表 maskn 中的每一行的第二列开始到最后一列的部分赋值给列表 masks 中的第四个元素（每一行的第一列开始到倒数第二列的部分）
                        masks2 = [maskn-e for e in masks] #4x512x512，masks2[0]是maskn的后一个元素减掉前一个元素（第0个元素前面没得减，所以他减掉的是1），masks2[1]是maskn的前一个元素减掉后一个元素（最后一个元素后面没得减，所以他减掉的是1），masks2[2]是maskn的每一行的后一个元素减掉前一个元素，masks2[3]是maskn的每一行的前一个元素减掉后一个元素，如果是边界位置，那么相减之后可能是-1或1，如果不是边界，那么相减之后就是0
                        bound = np.minimum.reduce(masks2)# 在axis=0维度上找到最小值，最后shape变为512x512，就是有边界的位置会是-1，没有边界就是0
                        bound = -bound #把数值从0到-1变到0到1
                        xb = []
                        yb = []
                        for i in range(4): # 对于每个局部区域，左眼、右眼、鼻子、嘴巴
                            xbi = [center[i,0]-rws[i]/2, center[i,0]+rws[i]/2-1] # 宽度的两个端点在:左端点和右端点
                            ybi = [center[i,1]-rhs[i]/2, center[i,1]+rhs[i]/2-1] # 高度的两个端点：上端点和下端点
                            for j in range(2):
                                maskx = bound[:,int(xbi[j])]# 在左端点或右端点的那一列的数据,有边界的位置是-1，没有边界的位置是0，一共512个数
                                masky = bound[int(ybi[j]),:] # 在上端点或下端点的那一行的数据 512
                                tmp_a = torch.from_numpy(maskx)*xbi[j].double()# 转为tensor，并乘以左端点或右端点的值，tmp_a有边界的位置变成特定值，没有边界的位置是0
                                tmp_b = torch.from_numpy(1-maskx)# tmp_b有边界的位置是0，没有边界的位置是1，tmp_b与tmp_a是互补的
                                xb += [tmp_b*10000 + tmp_a]#有边界的位置是特定值（就是局部区域端点的x坐标），没有边界的位置是10000

                                tmp_a = torch.from_numpy(masky)*ybi[j].double()
                                tmp_b = torch.from_numpy(1-masky)
                                yb += [tmp_b*10000 + tmp_a] #   有边界的位置是特定值（就是局部区域端点的y坐标），没有边界的位置是10000
                        soft = 1-getSoft([imgsize,imgsize],xb,yb) #xb和yb都是长度为8的list，每个list都512长度，边界位置是特定值，非边界位置是10000，经过getSoft处理之后，距离边界位置大于(boundwidth+1)的位置是1，小于(boundwidth+1)的位置是大于0小于1的数字，然后被1减之后，远离边界的位置变成0，越靠近边界的位置越接近1
                        soft = torch.Tensor(soft).unsqueeze(0)#soft是1x512x512,远离边界的位置是0，越靠近边界的位置越接近1
                        mask = (torch.ones(mask.shape)-mask)*soft + mask#此前mask眼睛鼻子嘴巴位置设置为0，其他位置还是1，被1减掉之后眼睛鼻子嘴巴位置为1，其他位置为0，然后乘以soft（远离边界位置为0，越靠近边界越接近1），得到的结果就是眼睛鼻子嘴巴靠近边界的位置接近1，远离边界的位置为0，而其他位置也都是0（不管是否靠近边界），再加上mask后，眼睛鼻子嘴巴外面都是1，里面靠近边界的地方越靠近边界越接近1，越远离边界越接近0，甚至等于0
                elif self.opt.region_enm == 1:
                    for i in range(4):
                        cmask0 = cmasks0[i]
                        rec = torch.zeros(B.shape)
                        rec[:,center[i,1]-rhs[i]/2:center[i,1]+rhs[i]/2,center[i,0]-rws[i]/2:center[i,0]+rws[i]/2] = 1
                        mask = mask * (torch.ones(B.shape) - cmask0 * rec)
                elif self.opt.region_enm == 2:
                    for i in range(4):
                        cmask0 = cmasks0[i]
                        mask = mask * (torch.ones(B.shape) - cmask0)
                hair_A = (A/2+0.5) * mask.repeat(int(input_nc/output_nc),1,1) * 2 - 1#A原来是-1到1，处理到0-1，然后再乘以mask（0-1），然后再处理到-1到1
                hair_B = (B/2+0.5) * mask * 2 - 1
                item['hair_A'] = hair_A # 3x512x512 经过软边界处理后的图像，数值在-1到1之间，在眼睛鼻子嘴巴区域越远离边界越接近-1
                item['hair_B'] = hair_B # 1x512x512
                item['mask'] = mask # mask out eyes, nose, mouth，这个mask数值在0-1之间，眼睛鼻子嘴巴外是1，里面越靠近边界越接近1，远离为0
                if self.opt.bg_local:
                    bgdir = self.opt.bg_dir
                    bgpath = os.path.join(bgdir,basen[:-4]+'.png')
                    im_bg = Image.open(bgpath) # 取出背景图来，是一个单通道的图像
                    mask2 = transforms.ToTensor()(im_bg) # mask out background，转化为tensor，数值在0-1之间，背景是0
                    if flipped:
                        mask2 = mask2.index_select(2, idx)
                    mask2 = (mask2 >= 0.5).float()# 二值化处理，>=0.5的设置为1，否则设置为0
                    hair_A = (A/2+0.5) * mask.repeat(int(input_nc//output_nc),1,1) * mask2.repeat(int(input_nc//output_nc),1,1) * 2 - 1#把背景从hair_A中去掉
                    hair_B = (B/2+0.5) * mask * mask2 * 2 - 1 #把背景从hair_B中去掉
                    bg_A = (A/2+0.5) * (torch.ones(mask2.shape)-mask2).repeat(int(input_nc//output_nc),1,1) * 2 - 1 #把背景取出来，没有用到软边界
                    bg_B = (B/2+0.5) * (torch.ones(mask2.shape)-mask2) * 2 - 1
                    item['hair_A'] = hair_A # 如果用到背景，那么还需要更新item字典中的hair_A
                    item['hair_B'] = hair_B
                    item['bg_A'] = bg_A
                    item['bg_B'] = bg_B
                    item['mask'] = mask # 软边界处理后的mask
                    item['mask2'] = mask2 # 没有经过软边界处理的mask
        
        if (self.opt.isTrain and self.opt.chamfer_loss):
            if self.opt.which_direction == 'AtoB':
                img = tocv2(B)
            else:
                img = tocv2(A)
            dt1, dt2 = dt(img)
            dt1 = torch.from_numpy(dt1) # [512,512]
            dt2 = torch.from_numpy(dt2)
            dt1 = dt1.unsqueeze(0) #[1,512,512]，每个像素到最近背景点的距离0-1
            dt2 = dt2.unsqueeze(0) #[1,512,512]，每个像素到最近背景点的距离0-1
            item['dt1gt'] = dt1
            item['dt2gt'] = dt2
        
        if self.opt.isTrain and self.opt.emphasis_conti_face:#constrain conti loss to pixels in original lines (avoid apply to background etc)
            face_mask_path = os.path.join(self.opt.facemask_dir,basen[:-4]+'.png')
            face_mask = Image.open(face_mask_path) # 打开脸部mask图像，是一个单通道的图像
            face_mask = transforms.ToTensor()(face_mask) # [0,1]
            if flipped:
                face_mask = face_mask.index_select(2, idx)
            item['face_mask'] = face_mask

        return item

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
