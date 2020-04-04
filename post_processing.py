# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import multiprocessing as mp

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data
    
def getDatasetDict(opt):
    df=pd.read_csv(opt["video_info"])
    json_data= load_json(opt["video_anno"])
    database=json_data
    video_dict={}
    for i in range(len(df)):
        video_name=df.video.values[i]
        video_info=database[video_name]
        video_new_info={}
        video_new_info['duration_frame']=video_info['duration_frame']
        video_new_info['duration_second']=video_info['duration_second']
        video_new_info["feature_frame"]=video_info['feature_frame']
        video_subset=df.subset.values[i]
        video_new_info['annotations']=video_info['annotations']
        if video_subset==opt["pem_inference_subset"]: # 只添加这个subset的video信息
            video_dict[video_name]=video_new_info
    return video_dict

def iou_with_anchors(anchors_min,anchors_max,len_anchors,box_min,box_max):
    """Compute jaccard score between a box and the anchors.
    """
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len +box_max-box_min
    #print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

def Soft_NMS(df,opt):
    df=df.sort_values(by="score",ascending=False)
    
    tstart=list(df.xmin.values[:])
    tend=list(df.xmax.values[:])
    tscore=list(df.score.values[:])
    rstart=[]
    rend=[]
    rscore=[]

    while len(tscore)>0 and len(rscore)<=opt["post_process_top_K"]:
        max_index=np.argmax(tscore)
        tmp_width = tend[max_index] -tstart[max_index]
        iou_list = iou_with_anchors(tstart[max_index],tend[max_index],tmp_width,np.array(tstart),np.array(tend))
        iou_exp_list = np.exp(-np.square(iou_list)/opt["soft_nms_alpha"])
        for idx in range(0,len(tscore)):
            if idx!=max_index:
                tmp_iou = iou_list[idx]
                if tmp_iou>opt["soft_nms_low_thres"] + (opt["soft_nms_high_thres"] - opt["soft_nms_low_thres"]) * tmp_width:
                    tscore[idx]=tscore[idx]*iou_exp_list[idx]
            
        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)
                
    newDf=pd.DataFrame()
    newDf['score']=rscore
    newDf['xmin']=rstart
    newDf['xmax']=rend
    return newDf

def video_post_process(opt,video_list,video_dict):

    for video_name in video_list:
        df=pd.read_csv("./output/PEM_results/"+video_name+".csv") # 读取见过
    
        df['score']=df.iou_score.values[:]*df.xmin_score.values[:]*df.xmax_score.values[:] # num_proposals score的融合
        if len(df)>1:
            df=Soft_NMS(df,opt) # 进行NMS
        
        df=df.sort_values(by="score",ascending=False) #分数的排序
        video_info=video_dict[video_name]
        video_duration=float(video_info["duration_frame"]/16*16)/video_info["duration_frame"]*video_info["duration_second"] #没看出来有啥用
        proposal_list=[]
    
        for j in range(min(opt["post_process_top_K"],len(df))):
            tmp_proposal={}
            tmp_proposal["score"]=df.score.values[j]
            tmp_proposal["segment"]=[max(0,df.xmin.values[j])*video_duration,min(1,df.xmax.values[j])*video_duration] # 将时间从归一化的时间映射到施水平片段上
            proposal_list.append(tmp_proposal)
        result_dict[video_name[2:]]=proposal_list # 添加到global变量中 {vide0_name: [{score:1, segments:[2, 3]},{score:1, segments:[1, 2]}]}}
        

def BSN_post_processing(opt):
    video_dict=getDatasetDict(opt) # 返回需要后处理的视频
    video_list=list(video_dict.keys())#[:100]
    del_videl_list = ['v_5HW6mjZZvtY']
    for v in del_videl_list:  # delete the video from video list, whose feature csv is broken
        if v in video_dict:
            print("del " + v + ' video')
            video_list.remove(v)
            del video_dict[v]
    global result_dict
    result_dict=mp.Manager().dict()
    
    num_videos = len(video_list)
    num_videos_per_thread = int(num_videos/opt["post_process_thread"])
    processes = []
    for tid in range(opt["post_process_thread"]-1): #多线程后处理
        tmp_video_list = video_list[tid*num_videos_per_thread:(tid+1)*num_videos_per_thread]
        p = mp.Process(target = video_post_process,args =(opt,tmp_video_list,video_dict,))
        p.start()
        processes.append(p)
    tmp_video_list = video_list[(opt["pgm_thread"]-1)*num_videos_per_thread:]
    p = mp.Process(target = video_post_process,args =(opt,tmp_video_list,video_dict,))
    p.start()
    processes.append(p)
    for p in processes:
        p.join()
    
    result_dict = dict(result_dict)
    output_dict={"version":"VERSION 1.3","results":result_dict,"external_data":{}} # 按照规则存储proposal的结果
    outfile=open(opt["result_file"],"w")
    json.dump(output_dict,outfile)
    outfile.close()

#opt = opts.parse_opt()
#opt = vars(opt)
#BSN_post_processing(opt)