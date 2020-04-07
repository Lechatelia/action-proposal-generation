# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
import math

            
class TEM(torch.nn.Module):
    def __init__(self, opt):
        super(TEM, self).__init__()
        
        self.feat_dim = opt["tem_feat_dim"] #提取的特征维度 默认是400 rgb+flow
        self.temporal_dim = opt["temporal_scale"] # 100 时间长度
        self.batch_size= opt["tem_batch_size"]
        self.c_hidden = opt["tem_hidden_dim"] # 隐藏层特征维度
        self.tem_best_loss = 10000000
        self.output_dim = 3  
        # 3个时域卷积，分别预测三个概率序列
        self.conv1 = torch.nn.Conv1d(in_channels=self.feat_dim,    out_channels=self.c_hidden,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv2 = torch.nn.Conv1d(in_channels=self.c_hidden,out_channels=self.c_hidden,kernel_size=3,stride=1,padding=1,groups=1)
        self.conv3 = torch.nn.Conv1d(in_channels=self.c_hidden,out_channels=self.output_dim,   kernel_size=1,stride=1,padding=0)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [bs, 400, 100]
        x = F.relu(self.conv2(x)) # [bs, 512, 100]
        x = torch.sigmoid(0.01*self.conv3(x)) #这个0.01的作用很大吗？？？
        return x

class PEM(torch.nn.Module):
    
    def __init__(self,opt):
        super(PEM, self).__init__()
        
        self.feat_dim = opt["pem_feat_dim"]
        self.batch_size = opt["pem_batch_size"]
        self.hidden_dim = opt["pem_hidden_dim"]
        self.u_ratio_m = opt["pem_u_ratio_m"]
        self.u_ratio_l = opt["pem_u_ratio_l"]
        self.output_dim = 1
        self.pem_best_loss = 1000000
        # 两个全连接网络 评估proposal的质量
        self.fc1 = torch.nn.Linear(in_features=self.feat_dim,out_features=self.hidden_dim,bias =True)
        self.fc2 = torch.nn.Linear(in_features=self.hidden_dim,out_features=self.output_dim,bias =True)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            #init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        x = F.relu(0.1*self.fc1(x))
        x = torch.sigmoid(0.1*self.fc2(x))
        return x # [num_proposal,1]


class BMN(nn.Module):
    def __init__(self, opt):
        super(BMN, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.prop_boundary_ratio = opt["bmn_prop_boundary_ratio"]
        self.num_sample = opt["bmn_num_sample"]
        self.num_sample_perbin = opt["bmn_num_sample_perbin"]
        self.feat_dim=opt["bmn_feat_dim"]

        self.hidden_dim_1d = 256
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask() # 得到一维插值的mask

        # Base Module
        self.x_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True)
        )

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        self.x_1d_p = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        base_feature = self.x_1d_b(x) # bs, 256, 100
        start = self.x_1d_s(base_feature).squeeze(1) # bs, 100
        end = self.x_1d_e(base_feature).squeeze(1) # bs, 100
        confidence_map = self.x_1d_p(base_feature) # bs, 256, 100
        confidence_map = self._boundary_matching_layer(confidence_map) # bs, 256, 32, 100, 100
        confidence_map = self.x_3d_p(confidence_map).squeeze(2) #[bs, 256, 100, 100]
        confidence_map = self.x_2d_p(confidence_map) # [bs, 2, 256, 256]
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size() # 2, 256, 100
        # 矩阵乘法  插值 mask [bs, 256, T=100]*[T=100, 32*D*T=320000] 得到的实际上是bm feature map
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale) # bs, 256, 32, D, T
        return out

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair 用于给一个candidate proposals产生插值mask
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0) # 每一个采样范围大小 实际上是采样了32*3个小区间 采样3个小区间而不是单纯的1个小区间线性插值是为了减小误差，
        # 相当于取周围三个点，这三个点都是通过1D插值得到的，然后均值其mask weights即可
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ] # 每一个采样点位置
        p_mask = []
        for idx in range(num_sample): #0到32个采样mask weight 每一个proposal的时间维度通过插值要归一化到32
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale]) # 先用0初始化这一行mask weight
            for sample in bin_samples:
                sample_upper = math.ceil(sample) # 大于这个数的第一个整数
                sample_decimal, sample_down = math.modf(sample) # 整数部分 与 小数部分
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal # 下采样点的mask 等于1-小数部分
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal # 上采样点的mask 等于小数部分
            bin_vector = 1.0 / num_sample_perbin * bin_vector # 因为采用了三个点的均值，所以1/3 [100]
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1) #[100, 32]
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map 用于产生BM feature map
        mask_mat = []
        for start_index in range(self.tscale):
            mask_mat_vector = []
            for duration_index in range(self.tscale): # 对于BM map上(i,j)个点都要生成一个32*T的map i=duration_index j=start_index
                if start_index + duration_index < self.tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio # 采样区域应该也包括周围部分
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask( # 对于这个点产生1d插值的 mask
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin) #[100, 32]
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample]) # [100, 32]
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2) #[100, 32, D]
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3) #[100, 32, 100, 100] float64
        mask_mat = mask_mat.astype(np.float32) # to float32
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False) # [100, 32*D*T] 因为DT 都不改变，可以直接一直用，而不用动态生成


if __name__ == '__main__':
    # Test BMN model
    import opts
    opt = opts.parse_opt()
    opt = vars(opt)
    model=BMN(opt)
    input=torch.randn(2,400,100)
    a,b,c=model(input)
    print(a.shape,b.shape,c.shape)