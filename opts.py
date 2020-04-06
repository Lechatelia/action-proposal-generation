import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--module',
        type=str,
        default='BSN_Evaluation')
        # default='BMN_Evaluation')
        # default='BMN_Post_processing')
        # default='PEM')
        # default='BMN')
    parser.add_argument(
        '--mode',
        type=str,
        # default='train')
        default='inference')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='./checkpoint')
        
    # Overall Dataset settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/activitynet_annotations/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/activitynet_annotations/anet_anno_action.json")
    
    # TEM Dataset settings
    parser.add_argument( # 时间尺度大小，相当于做了归一化，有技巧性在里面
        '--temporal_scale',
        type=int,
        default=100)
    parser.add_argument(
        '--boundary_ratio',
        type=float,
        default=0.1)
    parser.add_argument( # 存放特征的维度
        '--feature_path',
        type=str,
        default="./data/activitynet_feature_cuhk/")

    # BMN Dataset settings
    parser.add_argument(
        '--bmn_num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--bmn_num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--bmn_prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--bmn_feat_dim',
        type=int,
        default=400)

    # PEM Dataset settings
    parser.add_argument(
        '--pem_top_K',
        type=int,
        default=500)
    parser.add_argument(
        '--pem_top_K_inference',
        type=int,
        default=1000)

    # TEM model settings
    parser.add_argument( # 输入特征维度 200+200
        '--tem_feat_dim',
        type=int,
        default=400)
    parser.add_argument( #隐藏层维度
        '--tem_hidden_dim',
        type=int,
        default=512)


    # PEM model settings
    parser.add_argument(  # BSP特征维度
        '--pem_feat_dim',
        type=int,
        default=32)
    parser.add_argument(  # 全连接网络结构
        '--pem_hidden_dim',
        type=int,
        default=256)
    
    # TEM Training settings
    parser.add_argument(
        '--tem_training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--tem_weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--tem_epoch',
        type=int,
        default=20)
    parser.add_argument(
        '--tem_step_size',
        type=int,
        default=7)
    parser.add_argument(
        '--tem_step_gamma',
        type=float,
        default=0.1)
    parser.add_argument(
        '--tem_batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--tem_match_thres',
        type=float,
        default=0.5)

    # PEM Training settings
    parser.add_argument(
        '--pem_training_lr',
        type=float,
        default=0.01)
    parser.add_argument( # L2 正则化
        '--pem_weight_decay',
        type=float,
        default=0.00001)
    parser.add_argument(
        '--pem_epoch',
        type=int,
        default=20)
    parser.add_argument(
        '--pem_step_size',
        type=int,
        default=10)
    parser.add_argument(
        '--pem_step_gamma',
        type=float,
        default=0.1)
    parser.add_argument(
        '--pem_batch_size',
        type=int,
        default=16)
        # default=1) # it is set to 1 when inference
    parser.add_argument(
        '--pem_u_ratio_m',
        type=float,
        default=1)
    parser.add_argument(
        '--pem_u_ratio_l',
        type=float,
        default=2)
    parser.add_argument(
        '--pem_high_iou_thres',
        type=float,
        default=0.6)
    parser.add_argument(
        '--pem_low_iou_thres',
        type=float,
        default=0.2)

    # BMN Training settings
    parser.add_argument(
        '--bmn_training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--bmn_weight_decay',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--bmn_train_epochs',
        type=int,
        default=9)
    parser.add_argument(
        '--bmn_batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--bmn_step_size',
        type=int,
        default=7)
    parser.add_argument(
        '--bmn_step_gamma',
        type=float,
        default=0.1)

    # PEM inference settings
    parser.add_argument(  #  如果要测试不同的subset，更改这里就可以，然后运行PEMinference以及后面步骤即可
        '--pem_inference_subset',
        type=str,
        default="validation")

    # PGM settings
    parser.add_argument(
        '--pgm_threshold',
        type=float,
        default=0.5)
    parser.add_argument(
        '--pgm_thread',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_start',
        type=int,
        default=8)
    parser.add_argument(
        '--num_sample_end',
        type=int,
        default=8)
    parser.add_argument(
        '--num_sample_action',
        type=int,
        default=16) # num_sample_start + end + action should equal to pem_feat_dim
    parser.add_argument(
        '--num_sample_interpld',
        type=int,
        default=3)
    parser.add_argument(
        '--bsp_boundary_ratio',
        type=float,
        default=0.2)

    # Post processing
    parser.add_argument(
        '--post_process_top_K',
        type=int,
        default=100)
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=32)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.75)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.65)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--result_file',
        type=str,
        default="./output/result_proposal.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="./output/evaluation_result.jpg")

    # BMN Post processing
    parser.add_argument(
        '--bmn_post_process_thread',
        type=int,
        default=32)
    parser.add_argument(
        '--bmn_soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--bmn_soft_nms_low_thres',
        type=float,
        default=0.5)
    parser.add_argument(
        '--bmn_soft_nms_high_thres',
        type=float,
        default=0.9)
    parser.add_argument(
        '--bmn_result_file',
        type=str,
        default="./output/bmn_result_proposal.json")
    parser.add_argument(
        '--bmn_save_fig_path',
        type=str,
        default="./output/bmn_evaluation_result.jpg")

    args = parser.parse_args()

    return args
