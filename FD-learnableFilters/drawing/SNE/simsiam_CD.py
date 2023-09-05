import torch
import torch.nn as nn
import torch.nn.functional as F
from lossFunctions import *
from Encoder import EncoderIT
from model.model_main import RegionalExtraction1, RegionalExtraction2


def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class codeDiscriminator(nn.Module):
    def __init__(self, args):
        super(codeDiscriminator, self).__init__()
        # 特征输入

        self.frame_num = args.frame_num
        self.filter_num = args.filter_num
        self.l1 = nn.Linear(self.filter_num * self.frame_num, self.filter_num * self.frame_num // 8)
        self.l2 = nn.Linear(self.filter_num * self.frame_num // 8, self.filter_num)
        self.l3 = nn.Linear(self.filter_num, 1)

        self.l1 = nn.utils.spectral_norm(self.l1)  # 谱归一化
        self.l2 = nn.utils.spectral_norm(self.l2)  # 谱归一化
        self.l3 = nn.utils.spectral_norm(self.l3)  # 谱归一化

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        h1 = self.l1(x)
        h1 = nn.LeakyReLU(0.1)(h1.squeeze())
        h2 = self.l2(h1)
        h2 = nn.LeakyReLU(0.1)(h2)
        output = self.l3(h2)
        return output


class projection_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=128, out_dim=64):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=128, out_dim=64):  # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimSiam(nn.Module):

    def __init__(self, args):
        super().__init__()

        # features
        self.encoder = EncoderIT(args)

        # encoder classifier

        # mlp
        self.projector = projection_MLP()
        self.predictor = prediction_MLP()

        # Loss
        self.SIM_loss = CMD()
        # 正交
        self.DIFF_loss = DiffLoss()
        # self.DiffLoss_norm = DiffLoss_norm()

        self.batch_size = args.batch_size
        self.Mu = 10
        self.LAMBDA = 1
        self.GAMMA = 100
        # 在这个部分，我们定义两种特征提取部分，我们认为，不同的情感区域应该由不同的Classifier进行特征提取。
        # 简单讲：
        # 一致性区域
        self.Encoder_I_Classifier = RegionalExtraction1(args)
        # 差异性区域
        self.Encoder_T_Classifier = RegionalExtraction2(args)
        self.batch_size = args.batch_size
        self.filter_num = args.filter_num
        self.frame_num = args.frame_num

    def forward(self, x1, x2):
        enc, pro, pre = self.encoder, self.projector, self.predictor

        # backbone two encoder
        """Online"""
        # Encoder_I
        # Encoder_T
        """Target"""
        # Encoder_I
        # Encoder_T
        fI_online, fT_online, sort_Loss_online = enc(x1)
        fI_target, fT_target, sort_Loss_target = enc(x2)  # 下分支 batch_size x filters x Features
        L_enc = sort_Loss_online + sort_Loss_target

        # Loss_ct 控制变与不变特征之间约束 同类相似
        # 两个不同的语音片段, 差异性应该是巨大的.
        # 彼此之间相近
        SIM_lossI = self.SIM_loss(fI_online, fI_target, 2)

        gaussianFeat = torch.randn((self.batch_size, self.filter_num, self.frame_num)).cuda()
        GL_lossO = self.SIM_loss(gaussianFeat, fI_online, 2)
        GL_lossT = self.SIM_loss(gaussianFeat, fI_target, 2)
        global_Loss = GL_lossT + GL_lossO
        """
        Encoder I/T 之间的距离必须是较大的,不能学习同一个东西.
        如何判定不同，使用正交！
        """
        DIFF_lossI = self.DIFF_loss(fI_online, fT_online)
        DIFF_lossT = self.DIFF_loss(fI_target, fT_target)
        # DIFF_loss_extra = self.DIFF_loss(zT_online, zT_target)

        # ------------------------------------------------------------
        zI_online = self.Encoder_I_Classifier(fI_online)
        zI_target = self.Encoder_I_Classifier(fI_target)

        zT_online = self.Encoder_T_Classifier(fT_online)
        zT_target = self.Encoder_T_Classifier(fT_target)
        # ----------------------------------------------------
        SIM_loss_zT = self.SIM_loss(zT_online, zT_target, 2)
        # -----------------------------------
        m, n = zI_target.shape
        gaussian = torch.randn((m, n)).cuda()
        # --------------------------------
        gl_zI = self.SIM_loss(gaussian, zI_online, 2)
        gl_zT = self.SIM_loss(gaussian, zI_target, 2)
        global_Loss_z = gl_zI + gl_zT

        DIFF_loss_zI = self.DIFF_loss(zI_online, zT_online)
        DIFF_loss_zT = self.DIFF_loss(zI_target, zT_target)

        # 隐藏向量间的约束 5 代表距离远一些
        L_IT_f = (SIM_lossI + global_Loss) + (DIFF_lossI + DIFF_lossT)
        # L_IT_z = (SIM_loss_zT + global_Loss_z) + (DIFF_loss_zI + DIFF_loss_zT)
        L_IT_z = SIM_loss_zT + (DIFF_loss_zI + DIFF_loss_zT)
        L_IT = L_IT_z + L_IT_f

        # 特征拼接 展平 还是融合.
        # 1
        up_one_features = sum([zI_online, zT_online]).view(self.batch_size, -1)
        # 2
        up_two_features = sum([zI_online, zT_target]).view(self.batch_size, -1)
        # 3
        down_one_features = sum([zI_target, zT_target]).view(self.batch_size, -1)
        # 4
        down_two_features = sum([zI_target, zT_online]).view(self.batch_size, -1)
        # ---------------------------------------------
        # up_one_features = torch.cat([zI_online, zT_online], dim=1).view(self.batch_size, -1)
        # up_two_features = torch.cat([zI_online, zT_target], dim=1).view(self.batch_size, -1)
        #
        # down_one_features = torch.cat([zI_target, zT_online], dim=1).view(self.batch_size, -1)
        # down_two_features = torch.cat([zI_target, zT_target], dim=1).view(self.batch_size, -1)

        # projector (Z)
        # predictor（P）
        # 1
        p_up_1, p_up_2 = pro(up_one_features), pro(up_two_features)
        # 2
        h_up_1, h_up_2 = pre(p_up_1), pre(p_up_2)

        # 3
        p_down_1, p_down_2 = pro(down_one_features), pro(down_two_features)
        # 4
        h_down_1, h_down_2 = pre(p_down_1), pre(p_down_2)

        # 变/不变之间特征的互相组合
        L_S1 = D(h_up_1, p_down_1, version='simplified') / 2 + D(h_down_1, p_up_1, version='simplified') / 2
        L_S2 = D(h_down_1, p_up_2, version='simplified') / 2 + D(h_up_2, p_down_1, version='simplified') / 2

        L_S3 = D(h_up_2, p_down_2, version='simplified') / 2 + D(h_up_2, p_down_2, version='simplified') / 2
        L_S4 = D(h_up_1, p_down_2, version='simplified') / 2 + D(h_down_2, p_up_1, version='simplified') / 2

        # contrastive loss
        # 1和3 之间才是对比学习的主要学习内容。2和4受到gan的影响比较好学。
        L_similarity = ((L_S1 + L_S3) + (L_S2 + L_S4))  # 对比学习的主要损失函数

        print("Similarity Loss", L_similarity.tolist())

        # Loss1  相似性Loss
        # Loss2  Encoder 向量之间的Loss
        # Loss3  滤波器参数排序Loss

        LossTotal = self.Mu * L_similarity + self.LAMBDA * L_IT + self.GAMMA * L_enc

        return {'loss': LossTotal, 'I_online': fI_online, 'I_target': fI_target}
