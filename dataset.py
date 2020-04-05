# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pandas
import numpy
import json
import torch.utils.data as data
import os
import torch

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        self.temporal_scale = opt["temporal_scale"] # 时域长度 归一化到100
        self.temporal_gap = 1. / self.temporal_scale # 每个snippt时间占比
        self.subset = subset # training validation or test
        self.mode = opt["mode"] # 'train' or 'test'
        self.feature_path = opt["feature_path"] # '特征存放位置'
        self.boundary_ratio = opt["boundary_ratio"] # 0.1 人为扩充boundary的区域长度占总长度的比率
        self.video_info_path = opt["video_info"]  # 存在视频信息的csv
        self.video_anno_path = opt["video_anno"] # 存放标记信息的csv
        self._getDatasetDict()
        self.check_csv()

    def check_csv(self):
        # 因为某些视频的特征可能不存在，或者遭到了损坏
        for video in self.video_list:
            if not os.path.exists(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video + ".csv"):
                print("video :{} feature csv is not existed".format(video))
                self.video_list.remove(video)
                del self.video_dict[video]
        # 删除已知的错误样本
        del_videl_list = ['v_5HW6mjZZvtY']
        for v in del_videl_list:
            if v in self.video_dict:
                print("del " + v +' video')
                self.video_list.remove(v)
                del  self.video_dict[v]

        print ("After check: csv \n %s subset video numbers: %d" %(self.subset,len(self.video_list)))
        
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database= load_json(self.video_anno_path)
        self.video_dict = {} # 存放一系列内容，包括gt
        for i in range(len(anno_df)):
            video_name=anno_df.video.values[i]
            video_info=anno_database[video_name]
            video_subset=anno_df.subset.values[i] # 读取该视频属于的子数据集 training validation or test
            if self.subset == "full": #全部都要
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info # 是需要的数据集样本添加到字典中
        self.video_list = list(self.video_dict.keys()) # 含有哪些video
        print ("Before check: csv \n %s subset video numbers: %d" %(self.subset,len(self.video_list)))

    def __getitem__(self, index):
        video_data,anchor_xmin,anchor_xmax = self._get_base_data(index)
        if self.mode == "train":
            match_score_action,match_score_start,match_score_end =  self._get_train_label(index,anchor_xmin,anchor_xmax)
            return video_data,match_score_action,match_score_start,match_score_end
        else:
            return index,video_data,anchor_xmin,anchor_xmax
        
    def _get_base_data(self,index):
        video_name=self.video_list[index]
        anchor_xmin=[self.temporal_gap*i for i in range(self.temporal_scale)] # 0.00 d到 0.99
        anchor_xmax=[self.temporal_gap*i for i in range(1,self.temporal_scale+1)] # 0.01到0.10
        try:
            video_df=pd.read_csv(self.feature_path+ "csv_mean_"+str(self.temporal_scale)+"/"+video_name+".csv") # 得到这个视频的特征
        except:
            print('Error in '+video_name+".csv")
        video_data = video_df.values[:,:]
        video_data = torch.Tensor(video_data) # 这个video的特征[100, 400]
        video_data = torch.transpose(video_data,0,1) #[400， 100] 便于时域的一维卷积操作
        video_data.float()
        return video_data,anchor_xmin,anchor_xmax
    
    def _get_train_label(self,index,anchor_xmin,anchor_xmax): # 相当于要生成3个概率序列的真值
        video_name=self.video_list[index]
        video_info=self.video_dict[video_name] # 包括duration_second duration_frame annotations and feature_frame 但是这个特征长度已经被归一化了
        video_frame=video_info['duration_frame']
        video_second=video_info['duration_second']
        feature_frame=video_info['feature_frame']
        corrected_second=float(feature_frame)/video_frame*video_second  #相当于校准时间 因为采用的滑动窗口形式进行提取特征，两个frame会存在一些差异
        video_labels=video_info['annotations']
    
        gt_bbox = []
        for j in range(len(video_labels)): #将时间归一化 0到1之间
            tmp_info=video_labels[j]
            tmp_start=max(min(1,tmp_info['segment'][0]/corrected_second),0)
            tmp_end=max(min(1,tmp_info['segment'][1]/corrected_second),0)
            gt_bbox.append([tmp_start,tmp_end])
            
        gt_bbox=np.array(gt_bbox)
        gt_xmins=gt_bbox[:,0]
        gt_xmaxs=gt_bbox[:,1]

        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(self.temporal_gap,self.boundary_ratio*gt_lens) # starting region 和 ending region的长度
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1) # starting region
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1) # ending region

        # anchors = np.stack((anchor_xmin, anchor_xmax), 1) # 代表每一个snippet的范围
        match_score_action=[]
        # 给每一个位置计算TEM的三个概率值，但是from 0 to 99 效率不高吧 这种方法生成会有大量的无效操作，特别是gt较少的时候，可以后期优化
        for jdx in range(len(anchor_xmin)):
            match_score_action.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_xmins,gt_xmaxs)))
        match_score_start=[]
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_start_bboxs[:,0],gt_start_bboxs[:,1])))
        match_score_end=[]
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(self._ioa_with_anchors(anchor_xmin[jdx],anchor_xmax[jdx],gt_end_bboxs[:,0],gt_end_bboxs[:,1])))
        match_score_action = torch.Tensor(match_score_action)
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        return match_score_action,match_score_start,match_score_end #3个长度为100的概率序列

    def _ioa_with_anchors(self,anchors_min,anchors_max,box_min,box_max):
        len_anchors=anchors_max-anchors_min
        int_xmin = np.maximum(anchors_min, box_min)
        int_xmax = np.minimum(anchors_max, box_max)
        inter_len = np.maximum(int_xmax - int_xmin, 0.)
        scores = np.divide(inter_len, len_anchors)
        return scores

    def _ioa(self, anchors, gts):
        len_anchors = anchors[:,1] - anchors[:,0]
        int_min = np.maximum(anchors[:,0],gts[:,0])
        int_max = np.minimum(anchors[:,1],gts[:,1])
        np.maximum(np.expand_dims(np.arange(1, 5), 1), np.arange(3))


    
    def __len__(self):
        return len(self.video_list)


class ProposalDataSet(data.Dataset):
    def __init__(self,opt,subset="train"):
        
        self.subset=subset
        self.mode = opt["mode"]
        if self.mode == "train": # 测试与前推时的样本数量是不一样的
            self.top_K = opt["pem_top_K"]
        else:
            self.top_K = opt["pem_top_K_inference"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self.feature_path = opt["feature_path"]  # '特征存放位置'
        self.temporal_scale = opt["temporal_scale"]  # 时域长度 归一化到100
        self._getDatasetDict()
        self.check_csv()

    def check_csv(self):
        # 因为某些视频的特征可能不存在，或者遭到了损坏
        for video in self.video_list:
            if not os.path.exists(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video + ".csv"):
                print("video :{} feature csv is not existed".format(video))
                self.video_list.remove(video)
                del self.video_dict[video]
        # 删除已知的错误样本
        del_videl_list = ['v_5HW6mjZZvtY']
        for v in del_videl_list:
            if v in self.video_dict:
                print("del " + v +' video')
                self.video_list.remove(v)
                del  self.video_dict[v]

        print ("After check: csv \n %s subset video numbers: %d" %(self.subset,len(self.video_list)))
        
    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path) #读取信息
        anno_database= load_json(self.video_anno_path) # 读取相关真值信息
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name=anno_df.video.values[i]
            video_info=anno_database[video_name]
            video_subset=anno_df.subset.values[i]
            if self.subset == "full":
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print ("%s subset video numbers: %d" %(self.subset,len(self.video_list)))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video_name = self.video_list[index]
        pdf=pandas.read_csv("./output/PGM_proposals/"+video_name+".csv") # 读取proposal
        pdf=pdf[:self.top_K]
        video_feature = numpy.load("./output/PGM_feature/" + video_name+".npy") # read BSP feature for proposals
        video_feature = video_feature[:self.top_K,:]
        #print len(video_feature),len(pdf)
        video_feature = torch.Tensor(video_feature)

        if self.mode == "train":
            video_match_iou = torch.Tensor(pdf.match_iou.values[:]) # choose IOU as gt  已经在TEM inference阶段计算好
            return video_feature,video_match_iou # [bs, 32] [bs]
        else:
            # 取得proposals的 starting location， ending location, starting score, ending score
            video_xmin =pdf.xmin.values[:]
            video_xmax =pdf.xmax.values[:]
            video_xmin_score = pdf.xmin_score.values[:]
            video_xmax_score = pdf.xmax_score.values[:]
            return video_feature,video_xmin,video_xmax,video_xmin_score,video_xmax_score



def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class BMN_VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.video_info_path = opt["video_info"]
        self.video_anno_path = opt["video_anno"]
        self._getDatasetDict()
        self.check_csv()
        self._get_match_map()

    def check_csv(self):
        # 因为某些视频的特征可能不存在，或者遭到了损坏
        for video in self.video_list:
            if not os.path.exists(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video + ".csv"):
                print("video :{} feature csv is not existed".format(video))
                self.video_list.remove(video)
                del self.video_dict[video]
        # 删除已知的错误样本
        del_videl_list = ['v_5HW6mjZZvtY']
        for v in del_videl_list:
            if v in self.video_dict:
                print("del " + v +' video')
                self.video_list.remove(v)
                del  self.video_dict[v]

        print ("After check: csv \n %s subset video numbers: %d" %(self.subset,len(self.video_list)))

    def _getDatasetDict(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def __getitem__(self, index):
        video_data = self._load_file(index) # video feature [400, 100]
        if self.mode == "train":
            # [D, T] [100, 100] [T=100] [T=100]
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data,confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _get_match_map(self):
        match_map = []
        for idx in range(self.temporal_scale):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.temporal_scale + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.temporal_scale)]
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.temporal_scale + 1)]

    def _load_file(self, index):
        video_name = self.video_list[index]
        video_df = pd.read_csv(self.feature_path + "csv_mean_" + str(self.temporal_scale) + "/" + video_name + ".csv")
        video_data = video_df.values[:, :]
        video_data = torch.Tensor(video_data)
        video_data = torch.transpose(video_data, 0, 1)
        video_data.float()
        return video_data

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.temporal_scale, self.temporal_scale])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

if __name__ == '__main__':
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    # test dataset for BMN network
    train_loader = torch.utils.data.DataLoader(BMN_VideoDataSet(opt, subset="train"),
                                               batch_size=opt["bmn_batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)
    for a,b,c,d in train_loader:
        print(a.shape,b.shape,c.shape,d.shape)
        break