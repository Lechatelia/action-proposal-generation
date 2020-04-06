# -*- coding: utf-8 -*-
import sys
sys.path.append('./Evaluation')
from Evaluation.eval_proposal import ANETproposal
import matplotlib.pyplot as plt
import numpy as np

def run_evaluation(ground_truth_filename, proposal_filename, 
                   max_avg_nr_proposals=100, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=False)
    anet_proposal.evaluate()
    
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    
    return (average_nr_proposals, average_recall, recall)

def plot_metric(opt,average_nr_proposals, average_recall, recall,
                tiou_thresholds=np.linspace(0.5, 0.95, 10),save_fig_path="./output/evaluation_result.jpg"):

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1,1,1)
    
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]): #为每一个计算AUC
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]): # 绘制 0.5 0.6 0.7 0.8 0.9的ar曲线
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.),
                #the first 100 is max_avg_number_of_proposals, the seconds 100. is for the percentage presentation.
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0], # 绘制均值ar曲线
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    #plt.show()    
    plt.savefig(save_fig_path)

def evaluation_proposal(opt, result_file="./output/result_proposal.json", save_fig_path="./output/evaluation_result.jpg"):
    
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
        "./Evaluation/data/activity_net_1_3_new.json",
        result_file,
        max_avg_nr_proposals=100, # AN的最高限制 activitynet官方也是最大这个限制
        tiou_thresholds=np.linspace(0.5, 0.95, 10), # 插值十个点
        subset='validation')
    # 返回 average number of proposals， average recall， 不同阈值下面的recall
    # [100] [100] [10, 100] 这个100等于max_avg_nr_proposals
    plot_metric(opt,uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid, save_fig_path=save_fig_path)
    print ("AR@1 is \t",np.mean(uniform_recall_valid[:,0]))
    print ("AR@5 is \t",np.mean(uniform_recall_valid[:,4]))
    print ("AR@10 is \t",np.mean(uniform_recall_valid[:,9]))
    print ("AR@100 is \t",np.mean(uniform_recall_valid[:,-1]))