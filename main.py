import sys

sys.dont_write_bytecode = True
import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.optim as optim
import numpy as np
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import opts
from dataset import VideoDataSet, ProposalDataSet, BMN_VideoDataSet
from models import TEM, PEM, BMN
from loss_function import TEM_loss_function, PEM_loss_function
from bmn_loss_function import bmn_loss_func, get_mask
import pandas as pd
from pgm import PGM_proposal_generation, PGM_feature_generation
from post_processing import BSN_post_processing
from bmn_post_processing import BMN_post_processing
from eval import evaluation_proposal


def train_TEM(data_loader, model, optimizer, epoch, writer, opt):
    model.train()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter, (input_data, label_action, label_start, label_end) in enumerate(data_loader):
        TEM_output = model(input_data.cuda())  # [bs, 3, 100]
        loss = TEM_loss_function(label_action, label_start, label_end, TEM_output, opt)
        cost = loss["cost"]  # 得到损失函数

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 这里可以进行优化一下，每一个iter都进行输出
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()

    # 每一个epoch 显示一下损失函数的值
    writer.add_scalars('data/action', {'train': epoch_action_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/start', {'train': epoch_start_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/end', {'train': epoch_end_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/cost', {'train': epoch_cost / (n_iter + 1)}, epoch)

    print("TEM training loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" % (
    epoch, epoch_action_loss / (n_iter + 1),
    epoch_start_loss / (n_iter + 1),
    epoch_end_loss / (n_iter + 1)))


def test_TEM(data_loader, model, epoch, writer, opt):
    model.eval()
    epoch_action_loss = 0
    epoch_start_loss = 0
    epoch_end_loss = 0
    epoch_cost = 0
    for n_iter, (input_data, label_action, label_start, label_end) in enumerate(data_loader):
        TEM_output = model(input_data.cuda())
        loss = TEM_loss_function(label_action, label_start, label_end, TEM_output, opt)
        epoch_action_loss += loss["loss_action"].cpu().detach().numpy()
        epoch_start_loss += loss["loss_start"].cpu().detach().numpy()
        epoch_end_loss += loss["loss_end"].cpu().detach().numpy()
        epoch_cost += loss["cost"].cpu().detach().numpy()

    writer.add_scalars('data/action', {'test': epoch_action_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/start', {'test': epoch_start_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/end', {'test': epoch_end_loss / (n_iter + 1)}, epoch)
    writer.add_scalars('data/cost', {'test': epoch_cost / (n_iter + 1)}, epoch)

    print("TEM testing  loss(epoch %d): action - %.03f, start - %.03f, end - %.03f" % (
    epoch, epoch_action_loss / (n_iter + 1),
    epoch_start_loss / (n_iter + 1),
    epoch_end_loss / (n_iter + 1)))
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/tem_checkpoint.pth.tar")
    if epoch_cost < model.module.tem_best_loss:
        model.module.tem_best_loss = np.mean(epoch_cost)
        torch.save(state, opt["checkpoint_path"] + "/tem_best.pth.tar")


def train_PEM(data_loader, model, optimizer, epoch, writer, opt):
    model.train()
    epoch_iou_loss = 0

    for n_iter, (input_data, label_iou) in enumerate(data_loader):
        PEM_output = model(input_data.cuda())
        iou_loss = PEM_loss_function(PEM_output, label_iou, model, opt)  # loss smooth_l1_loss
        optimizer.zero_grad()
        iou_loss.backward()
        optimizer.step()
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'train': epoch_iou_loss / (n_iter + 1)}, epoch)

    print("PEM training loss(epoch %d): iou - %.04f" % (epoch, epoch_iou_loss / (n_iter + 1)))


def test_PEM(data_loader, model, epoch, writer, opt):
    model.eval()
    epoch_iou_loss = 0

    for n_iter, (input_data, label_iou) in enumerate(data_loader):
        PEM_output = model(input_data.cuda())
        iou_loss = PEM_loss_function(PEM_output, label_iou, model, opt)
        epoch_iou_loss += iou_loss.cpu().detach().numpy()

    writer.add_scalars('data/iou_loss', {'validation': epoch_iou_loss / (n_iter + 1)}, epoch)

    print("PEM testing  loss(epoch %d): iou - %.04f" % (epoch, epoch_iou_loss / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/pem_checkpoint.pth.tar")
    if epoch_iou_loss < model.module.pem_best_loss:
        model.module.pem_best_loss = np.mean(epoch_iou_loss)
        torch.save(state, opt["checkpoint_path"] + "/pem_best.pth.tar")


def BSN_Train_TEM(opt):
    writer = SummaryWriter()
    model = TEM(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt["tem_training_lr"], weight_decay=opt["tem_weight_decay"])

    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=model.module.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=model.module.batch_size, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=True)
    # 控制学习率
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["tem_step_size"], gamma=opt["tem_step_gamma"])

    for epoch in range(opt["tem_epoch"]):
        scheduler.step()
        train_TEM(train_loader, model, optimizer, epoch, writer, opt)  # 网络训练
        test_TEM(test_loader, model, epoch, writer, opt)  # 网络测试
    writer.close()


def BSN_Train_PEM(opt):
    writer = SummaryWriter()
    model = PEM(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt["pem_training_lr"], weight_decay=opt["pem_weight_decay"])

    def collate_fn(batch):  # 如何收集一个batch的数据
        batch_data = torch.cat([x[0] for x in batch])  # 在第一个维度上面直接拼接 [num_proposals, 32]
        batch_iou = torch.cat([x[1] for x in batch])  # num_proposals
        return batch_data, batch_iou

    train_loader = torch.utils.data.DataLoader(ProposalDataSet(opt, subset="train"),
                                               batch_size=model.module.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt, subset="validation"),
                                              batch_size=model.module.batch_size, shuffle=True,
                                              num_workers=8, pin_memory=True, drop_last=True, collate_fn=collate_fn)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["pem_step_size"], gamma=opt["pem_step_gamma"])

    for epoch in range(opt["pem_epoch"]):
        scheduler.step()
        train_PEM(train_loader, model, optimizer, epoch, writer, opt)  # PEM训练
        test_PEM(test_loader, model, epoch, writer, opt)  # 测试

    writer.close()


def BSN_inference_TEM(opt):
    model = TEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"] + "/tem_best.pth.tar")  # 装载模型
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="full"),
                                              batch_size=model.module.batch_size, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)

    columns = ["action", "start", "end", "xmin", "xmax"]
    # 主要是将各个视频的三个概率序列给存储起来，便于后续模块生成proposal
    for index_list, input_data, anchor_xmin, anchor_xmax in test_loader:

        TEM_output = model(input_data.cuda()).detach().cpu().numpy()
        batch_action = TEM_output[:, 0, :]
        batch_start = TEM_output[:, 1, :]
        batch_end = TEM_output[:, 2, :]

        index_list = index_list.numpy()
        anchor_xmin = np.array([x.numpy()[0] for x in anchor_xmin])
        anchor_xmax = np.array([x.numpy()[0] for x in anchor_xmax])

        for batch_idx, full_idx in enumerate(index_list):
            video = test_loader.dataset.video_list[full_idx]  # video name
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]  # 三个概率序列
            # 拼接起来，最后两列是时间帧 比如0.00 到 0.01
            video_result = np.stack((video_action, video_start, video_end, anchor_xmin, anchor_xmax), axis=1)
            video_df = pd.DataFrame(video_result, columns=columns)
            video_df.to_csv("./output/TEM_results/" + video + ".csv", index=False)


def BSN_inference_PEM(opt):
    model = PEM(opt)
    checkpoint = torch.load(opt["checkpoint_path"] + "/pem_best.pth.tar")  # 装载模型
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
    model.load_state_dict(base_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.eval()

    test_loader = torch.utils.data.DataLoader(ProposalDataSet(opt, subset=opt["pem_inference_subset"]),
                                              # batch_size=model.module.batch_size, shuffle=False,
                                              batch_size=1, shuffle=False,  # bs is set to 1
                                              num_workers=8, pin_memory=True, drop_last=False)

    for idx, (video_feature, video_xmin, video_xmax, video_xmin_score, video_xmax_score) in enumerate(test_loader):
        video_name = test_loader.dataset.video_list[idx]
        video_conf = model(video_feature.cuda()).view(-1).detach().cpu().numpy()  # 得到评估分数
        video_xmin = video_xmin.view(-1).cpu().numpy()
        video_xmax = video_xmax.view(-1).cpu().numpy()
        video_xmin_score = video_xmin_score.view(-1).cpu().numpy()
        video_xmax_score = video_xmax_score.view(-1).cpu().numpy()

        df = pd.DataFrame()  # 存储proposals的结果，分别对应着三个分数，starting/ending location scores, evaluation score from PEM
        df["xmin"] = video_xmin
        df["xmax"] = video_xmax
        df["xmin_score"] = video_xmin_score
        df["xmax_score"] = video_xmax_score
        df["iou_score"] = video_conf

        df.to_csv("./output/PEM_results/" + video_name + ".csv", index=False)


def train_BMN(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()
        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())
        optimizer.zero_grad()
        loss[0].backward()
        optimizer.step()

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))


def test_BMN(data_loader, model, epoch, bm_mask):
    model.eval()
    best_loss = 1e10
    epoch_pemreg_loss = 0
    epoch_pemclr_loss = 0
    epoch_tem_loss = 0
    epoch_loss = 0
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        input_data = input_data.cuda()
        label_start = label_start.cuda()
        label_end = label_end.cuda()
        label_confidence = label_confidence.cuda()

        confidence_map, start, end = model(input_data)
        loss = bmn_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        epoch_pemreg_loss += loss[2].cpu().detach().numpy()
        epoch_pemclr_loss += loss[3].cpu().detach().numpy()
        epoch_tem_loss += loss[1].cpu().detach().numpy()
        epoch_loss += loss[0].cpu().detach().numpy()

    print(
        "BMN training loss(epoch %d): tem_loss: %.03f, pem class_loss: %.03f, pem reg_loss: %.03f, total_loss: %.03f" % (
            epoch, epoch_tem_loss / (n_iter + 1),
            epoch_pemclr_loss / (n_iter + 1),
            epoch_pemreg_loss / (n_iter + 1),
            epoch_loss / (n_iter + 1)))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["checkpoint_path"] + "/BMN_checkpoint.pth.tar")
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(state, opt["checkpoint_path"] + "/BMN_best.pth.tar")


def BMN_Train(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["bmn_training_lr"],
                           weight_decay=opt["bmn_weight_decay"])

    train_loader = torch.utils.data.DataLoader(BMN_VideoDataSet(opt, subset="train"),
                                               batch_size=opt["bmn_batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(BMN_VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["bmn_batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["bmn_step_size"], gamma=opt["bmn_step_gamma"])
    bm_mask = get_mask(opt["temporal_scale"])
    for epoch in range(opt["bmn_train_epochs"]):
        scheduler.step()
        train_BMN(train_loader, model, optimizer, epoch, bm_mask)
        test_BMN(test_loader, model, epoch, bm_mask)


def BMN_inference(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/BMN_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(BMN_VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            # print(start.shape,end.shape,confidence_map.shape)
            start_scores = start[0].detach().cpu().numpy()
            end_scores = end[0].detach().cpu().numpy()
            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            max_start = max(start_scores)
            max_end = max(end_scores)

            ####################################################################################################
            # generate the set of start points and end points
            start_bins = np.zeros(len(start_scores))
            start_bins[0] = 1  # [1,0,0...,0,1] 首末两帧
            for idx in range(1, tscale - 1):
                if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
                    start_bins[idx] = 1
                elif start_scores[idx] > (0.5 * max_start):
                    start_bins[idx] = 1

            end_bins = np.zeros(len(end_scores))
            end_bins[-1] = 1
            for idx in range(1, tscale - 1):
                if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
                    end_bins[idx] = 1
                elif end_scores[idx] > (0.5 * max_end):
                    end_bins[idx] = 1
            ########################################################################################################

            #########################################################################
            # 遍历起始分界点与结束分界点的组合
            new_props = []
            for idx in range(tscale):
                for jdx in range(tscale):
                    start_index = jdx
                    end_index = start_index + idx + 1
                    if end_index < tscale and start_bins[start_index] == 1 and end_bins[end_index] == 1:
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        xmin_score = start_scores[start_index]
                        xmax_score = end_scores[end_index]
                        clr_score = clr_confidence[idx, jdx]
                        reg_score = reg_confidence[idx, jdx]
                        score = xmin_score * xmax_score * clr_score * reg_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./output/BMN_results/" + video_name + ".csv", index=False)


def main(opt):
    if opt["module"] == "TEM":
        if opt["mode"] == "train":
            print("TEM training start")
            BSN_Train_TEM(opt)
            print("TEM training finished")
        elif opt["mode"] == "inference":
            print("TEM inference start")
            if not os.path.exists("output/TEM_results"):
                os.makedirs("output/TEM_results")
            BSN_inference_TEM(opt)  # 前推产生每一个video的三个概率序列 并存储到TEM-results
            print("TEM inference finished")
        else:
            print("Wrong mode. TEM has two modes: train and inference")

    elif opt["module"] == "PGM":
        if not os.path.exists("output/PGM_proposals"):
            os.makedirs("output/PGM_proposals")
        print("PGM: start generate proposals")
        PGM_proposal_generation(opt)  # using staring and ending possibility produced by TEM to generate proposal
        print("PGM: finish generate proposals")

        if not os.path.exists("output/PGM_feature"):
            os.makedirs("output/PGM_feature")  #
        print("PGM: start generate BSP feature")
        PGM_feature_generation(opt)  # generate BSP features for all candidate proposals and store them
        print("PGM: finish generate BSP feature")

    elif opt["module"] == "PEM":
        if opt["mode"] == "train":
            print("PEM training start")
            BSN_Train_PEM(opt)  # train PEM module
            print("PEM training finished")
        elif opt["mode"] == "inference":
            if not os.path.exists("output/PEM_results"):
                os.makedirs("output/PEM_results")
            print("PEM inference start")
            BSN_inference_PEM(opt)  # evaluate the proposal from one subset and store them
            print("PEM inference finished")
        else:
            print("Wrong mode. PEM has two modes: train and inference")

    elif opt["module"] == "BMN":
        if opt["mode"] == "train":
            print("BMN training start")
            BMN_Train(opt)
            print("BMN training start")
        elif opt["mode"] == "inference":
            if not os.path.exists("output/BMN_results"):
                os.makedirs("output/BMN_results")
            print("BMN inference start")
            BMN_inference(opt)
            print("BMN inference start")
        else:
            print("Wrong mode. PEM has two modes: train and inference")
            print("Post processing start")

    elif opt["module"] == "BMN_Post_processing":
        print("BMN Post processing start")
        BMN_post_processing(opt)
        print("BMN Post processing finished")

    elif opt["module"] == "BSN_Post_processing":
        print("BSN Post processing start")
        BSN_post_processing(opt)
        print("BSN Post processing finished")

    elif opt["module"] == "BMN_Evaluation":
        evaluation_proposal(opt, result_file=opt["bmn_result_file"], save_fig_path=opt["bmn_save_fig_path"])

    elif opt["module"] == "BSN_Evaluation":
        evaluation_proposal(opt, result_file=opt["result_file"], save_fig_path=opt["save_fig_path"])
    print("")


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(opt["checkpoint_path"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()
    main(opt)
