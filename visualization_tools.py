# @Time : 2020/4/15 22:25 
# @Author : Jinguo Zhu
# @File : visualization_tools.py 
# @Software: PyCharm
'''
this used for visualizing the prediction

 '''
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import json


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class Visualizer():
    def __init__(self, subset="full"):
        self.prediction_path = "./output/TEM_results/"
        self.subset = subset
        self.temporal_scale = 100
        self.feature_path = "./data/activitynet_feature_cuhk/"
        self.video_info_path = "./data/activitynet_annotations/video_info_new.csv"
        self.video_anno_path = "./data/activitynet_annotations/anet_anno_action.json"
        self.proposal_result = "./output/result_proposal.json"
        self.get_gt()
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
                print("del " + v + ' video')
                self.video_list.remove(v)
                del self.video_dict[v]

        print("After check: csv \n %s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def get_gt(self):
        anno_df = pd.read_csv(self.video_info_path)
        anno_database = load_json(self.video_anno_path)
        self.video_dict = {}  # 存放一系列内容，包括gt
        for i in range(len(anno_df)):
            video_name = anno_df.video.values[i]
            video_info = anno_database[video_name]
            video_subset = anno_df.subset.values[i]  # 读取该视频属于的子数据集 training validation or test
            if self.subset == "full":  # 全部都要
                self.video_dict[video_name] = video_info
            if self.subset in video_subset:
                self.video_dict[video_name] = video_info  # 是需要的数据集样本添加到字典中
        self.video_list = list(self.video_dict.keys())  # 含有哪些video
        print("Before check: csv \n %s subset video numbers: %d" % (self.subset, len(self.video_list)))

    def get_prediction(self, video_name):
        pdf = pd.read_csv(os.path.join(self.prediction_path, video_name + ".csv"))
        return pdf.action.values, pdf.start.values, pdf.end.values, pdf.xmin.values, pdf.xmax.values

    def visualize(self, index):

        index = index % len(self.video_list)
        video_name = self.video_list[index]
        print("visualize {}".format(video_name))
        video_info = self.video_dict[
            video_name]  # 包括duration_second duration_frame annotations and feature_frame 但是这个特征长度已经被归一化了
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # 相当于校准时间 因为采用的滑动窗口形式进行提取特征，两个frame会存在一些差异
        video_labels = video_info['annotations']

        gt_bbox = []
        for j in range(len(video_labels)):  # 将时间归一化 0到1之间
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
        if len(gt_bbox) == 0:
            gt_bbox.append([0.0, 0.0])
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]

        action, start, end, xmin, xmax = self.get_prediction(video_name)

        # plot  action
        # self.visual_action(action, (xmin + xmax) / 2, gt_bbox)
        colors = ['r', 'teal', "green"]
        self.visual_actions(action, start, end, (xmin + xmax) / 2, gt_bbox, colors=colors)
        # self.visual_3_action(action, start, end, (xmin + xmax) / 2, gt_bbox, colors=colors)
        self.visual_3_action_v2(action, start, end, (xmin + xmax) / 2, gt_bbox, colors=colors)

    def visual_rand(self):
        for i in range(len(self.video_list)):
            self.visualize(i)

    def visual_action(self, action_score, time, gts):
        # plt.figure(figsize=(100,10))
        for gt in gts:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)

        plt.plot(time, action_score, "r")
        plt.xlabel('time')
        plt.ylabel('action')
        plt.show()

    def visual_actions(self, action_score, start, end, time, gts, colors):
        plt.figure(figsize=(8,2))
        for gt in gts:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)

        actions = [action_score, start, end]
        for i, color in enumerate(colors):
            plt.plot(time, actions[i], color=color)
        plt.xlabel('time')
        plt.ylabel('action')
        plt.show()

    def visual_3_action(self, action_score, start, end, time, gts, colors):
        # plt.figure(figsize=(100,10))
        # actions = [action_score, start, end]
        plt.subplot(311)
        for gt in gts:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)
        plt.plot(time, action_score, color=colors[0], label='action')
        # plt.xlabel('time')
        plt.ylabel('action')

        plt.subplot(312)
        for gt in gts:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)
        plt.plot(time, start, color=colors[1], label='start')
        # plt.xlabel('time')
        plt.ylabel('start')

        plt.subplot(313)
        for gt in gts:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)
        plt.plot(time, end, color=colors[2], label='end')
        plt.xlabel('time')
        plt.ylabel('end')


        plt.show()

    def visual_3_action_v2(self, action_score, start, end, time, gts, colors):
        # plt.figure(figsize=(100,10))
        # actions = [action_score, start, end]

        gt_xmins=gts[:,0]
        gt_xmaxs=gts[:,1]

        gt_lens=gt_xmaxs-gt_xmins
        gt_len_small=np.maximum(0.03,0.1*gt_lens) # starting region 和 ending region的长度
        gt_start_bboxs=np.stack((gt_xmins-gt_len_small/2,gt_xmins+gt_len_small/2),axis=1) # starting region
        gt_end_bboxs=np.stack((gt_xmaxs-gt_len_small/2,gt_xmaxs+gt_len_small/2),axis=1) # ending region

        plt.subplot(311)
        for gt in gts:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)
        plt.plot(time, action_score, color=colors[0], label='action')
        # plt.xlabel('time')
        plt.ylabel('action')

        plt.subplot(312)
        for gt in gt_start_bboxs:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)
        plt.plot(time, start, color=colors[1], label='start')
        # plt.xlabel('time')
        plt.ylabel('start')

        plt.subplot(313)
        for gt in gt_end_bboxs:
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, 0, 1, color="b", alpha=0.3, interpolate=True)
        plt.plot(time, end, color=colors[2], label='end')
        plt.xlabel('time')
        plt.ylabel('end')


        plt.show()

    def visual_proposals(self):
        self.proposals = load_json(self.proposal_result)['results']
        for i in range(len(self.video_list)):
            self.visualize(i)
            self.visualize_proposal(i)

    def visualize_proposal(self, index):
        index = index % len(self.video_list)
        video_name = self.video_list[index]
        print("visualize {}".format(video_name))

        video_info = self.video_dict[
            video_name]  # 包括duration_second duration_frame annotations and feature_frame 但是这个特征长度已经被归一化了
        video_frame = video_info['duration_frame']
        video_second = video_info['duration_second']
        feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # 相当于校准时间 因为采用的滑动窗口形式进行提取特征，两个frame会存在一些差异
        video_labels = video_info['annotations']

        proposals = self.proposals[video_name[2:]]
        result = []
        for proposal in proposals:
            # 时间归一化
            tmp_start = max(min(1, proposal['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, proposal['segment'][1] / corrected_second), 0)
            result.append([tmp_start, tmp_end, proposal['score']])
        result = np.array(result)
        # 按照分数排序
        result = result[np.lexsort(result.T)[::-1]] # [100, 3]

        gt_bbox = []
        for j in range(len(video_labels)):  # 将时间归一化 0到1之间
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
        if len(gt_bbox) == 0:
            gt_bbox.append([0.0, 0.0])
        gt_bbox = np.array(gt_bbox)

        colors = [ 'teal', "green", 'blueviolet', 'cyan', 'tomato','r']
        self.plot_proposals(result[:5,:], gt_bbox, colors)

    def plot_proposals(self, proposals, gts, colors):
        # proposal [N1, 3] start, end, score
        # gts [N2, 2] start, end
        # colors [N1+1]
        assert len(colors)==proposals.shape[0]+1
        # plt.xlim((0, 1))
        for  i in range(proposals.shape[0]):
            # plt.subplot(len(colors), 1, i+2)
            proposal = proposals[i]
            proposal_duration = np.linspace(proposal[0], proposal[1], 5)
            plt.fill_between(proposal_duration, len(proposals)-i-0.95, len(proposals)-i-0.05, color=colors[i],
                             alpha=0.3, interpolate=True)
            plt.text((proposal[0] + proposal[1])/2,  len(proposals)-i-0.8,"%.2f"%proposal[2],ha='center', va='bottom',fontsize=10)
        plt.xlabel('proposal')
        for gt in gts:
            # plt.subplot(len(colors), 1, len(colors))
            gt_duration = np.linspace(gt[0], gt[1], 10)
            plt.fill_between(gt_duration, len(proposals), len(proposals)+ 1, color=colors[-1],
                             alpha=0.3, interpolate=True, label='lable')
            plt.xlabel('label')
        # plt.ylabel('start')

        plt.show()





if __name__ == "__main__":
    # vis = Visualizer("full")
    # vis.visual_rand() # 用于三个概率序列的可视化
    # print("there are {} videos".format(len(vis.video_list)))

    # 用于最终proposal的可视化
    vis = Visualizer("validation") # should be opt["pem_inference_subset"]
    print("there are {} videos".format(len(vis.video_list)))
    vis.visual_proposals() # 用于最终proposals结果的可视化操作