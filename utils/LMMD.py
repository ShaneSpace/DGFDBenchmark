import torch
import torch.nn as nn
import numpy as np
# https://github.com/easezyc/deep-transfer-learning/blob/master/UDA/pytorch1.0/DSAN/lmmd.py
class LMMDLoss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, device = 'cpu', fix_sigma=None):
        super(LMMDLoss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type
        self.device = device

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_logits):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_logits, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).to(self.device)
        weight_tt = torch.from_numpy(weight_tt).to(self.device)
        weight_st = torch.from_numpy(weight_st).to(self.device)
        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).to(self.device)
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_weight(self, s_label, t_logits, batch_size=32, class_num=31):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_logits.cpu().data.max(1)[1].numpy()
        t_vec_label = t_logits.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        index = list(set(s_sca_label) & set(t_sca_label))
        mask_arr = np.zeros((batch_size, class_num))
        mask_arr[:, index] = 1
        t_vec_label = t_vec_label * mask_arr
        s_vec_label = s_vec_label * mask_arr

        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        weight_st = np.matmul(s_vec_label, t_vec_label.T)

        length = len(index)
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')

















































# import torch
# import torch.nn as nn
# import numpy as np

# class MMDLoss(nn.Module):
#     def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, **kwargs):
#         super(MMDLoss, self).__init__()
#         self.kernel_num = kernel_num
#         self.kernel_mul = kernel_mul
#         self.fix_sigma = None
#         self.kernel_type = kernel_type

#     def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
#         n_samples = int(source.size()[0]) + int(target.size()[0]) #总样本数n_s + n_t
#         total = torch.cat([source, target], dim=0)                #将source和target拼接起来
#         total0 = total.unsqueeze(0).expand(
#             int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         total1 = total.unsqueeze(1).expand(
#             int(total.size(0)), int(total.size(0)), int(total.size(1)))
#         L2_distance = ((total0-total1)**2).sum(2)
#         if fix_sigma:
#             bandwidth = fix_sigma
#         else:
#             bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
#         bandwidth /= kernel_mul ** (kernel_num // 2)
#         bandwidth_list = [bandwidth * (kernel_mul**i)
#                           for i in range(kernel_num)]
#         kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
#                       for bandwidth_temp in bandwidth_list]
#         return sum(kernel_val) #权重相等，这里5个不同bandwidth的来自一个family的核直接相加

#     def linear_mmd2(self, f_of_X, f_of_Y):
#         loss = 0.0
#         delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
#         loss = delta.dot(delta.T)
#         return loss

#     def forward(self, source, target):
#         if self.kernel_type == 'linear':
#             return self.linear_mmd2(source, target)
#         elif self.kernel_type == 'rbf':
#             batch_size = int(source.size()[0])
#             kernels = self.guassian_kernel(
#                 source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#             XX = torch.mean(kernels[:batch_size, :batch_size])
#             YY = torch.mean(kernels[batch_size:, batch_size:])
#             XY = torch.mean(kernels[:batch_size, batch_size:])
#             YX = torch.mean(kernels[batch_size:, :batch_size])
#             loss = torch.mean(XX + YY - XY - YX)
#             return loss

# class LambdaSheduler(nn.Module):
#     def __init__(self, gamma=1.0, max_iter=1000, **kwargs):
#         super(LambdaSheduler, self).__init__()
#         self.gamma = gamma
#         self.max_iter = max_iter
#         self.curr_iter = 0

#     def lamb(self):
#         p = self.curr_iter / self.max_iter
#         lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
#         return lamb

#     def step(self):
#         self.curr_iter = min(self.curr_iter + 1, self.max_iter)

# class LMMDLoss(MMDLoss, LambdaSheduler):
#     def __init__(self, num_class, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None,
#                     gamma=1.0, max_iter=1000, device='cpu', **kwargs):
#         '''
#         Local MMD
#         '''
#         super(LMMDLoss, self).__init__(kernel_type, kernel_mul, kernel_num, fix_sigma, **kwargs)
#         super(MMDLoss, self).__init__(gamma, max_iter, **kwargs)
#         self.num_class = num_class
#         self.device = device

#     def forward(self, source, target, source_label, target_logits):
#         if self.kernel_type == 'linear':
#             raise NotImplementedError("Linear kernel is not supported yet.")

#         elif self.kernel_type == 'rbf':
#             batch_size = source.size()[0]
#             weight_ss, weight_tt, weight_st = self.cal_weight(source_label, target_logits)
#             weight_ss = torch.from_numpy(weight_ss).to(self.device) # B, B
#             weight_tt = torch.from_numpy(weight_tt).to(self.device)
#             weight_st = torch.from_numpy(weight_st).to(self.device)

#             kernels = self.guassian_kernel(source, target,
#                                     kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
#             loss = torch.Tensor([0]).to(self.device)
#             if torch.sum(torch.isnan(sum(kernels))):
#                 return loss
#             SS = kernels[:batch_size, :batch_size]
#             TT = kernels[batch_size:, batch_size:]
#             ST = kernels[:batch_size, batch_size:]

#             loss += torch.sum( weight_ss * SS + weight_tt * TT - 2 * weight_st * ST )
#             # Dynamic weighting
#             lamb = self.lamb()
#             self.step()
#             loss = loss * lamb
#             return loss

#     def cal_weight(self, source_label, target_logits):
#         batch_size = source_label.size()[0]
#         source_label = source_label.cpu().data.numpy()
#         source_label_onehot = np.eye(self.num_class)[source_label] # one hot

#         source_label_sum = np.sum(source_label_onehot, axis=0).reshape(1, self.num_class)
#         source_label_sum[source_label_sum == 0] = 100
#         source_label_onehot = source_label_onehot / source_label_sum # label ratio

#         # Pseudo label
#         target_label = target_logits.cpu().data.max(1)[1].numpy()

#         target_logits = target_logits.cpu().data.numpy()
#         target_logits_sum = np.sum(target_logits, axis=0).reshape(1, self.num_class)
#         target_logits_sum[target_logits_sum == 0] = 100
#         target_logits = target_logits / target_logits_sum

#         weight_ss = np.zeros((batch_size, batch_size))
#         weight_tt = np.zeros((batch_size, batch_size))
#         weight_st = np.zeros((batch_size, batch_size))

#         set_s = set(source_label)
#         set_t = set(target_label)
#         count = 0
#         for i in range(self.num_class): # (B, C)
#             if i in set_s and i in set_t:
#                 s_tvec = source_label_onehot[:, i].reshape(batch_size, -1) # (B, 1)
#                 t_tvec = target_logits[:, i].reshape(batch_size, -1) # (B, 1)

#                 ss = np.dot(s_tvec, s_tvec.T) # (B, B)
#                 weight_ss = weight_ss + ss
#                 tt = np.dot(t_tvec, t_tvec.T)
#                 weight_tt = weight_tt + tt
#                 st = np.dot(s_tvec, t_tvec.T)
#                 weight_st = weight_st + st
#                 count += 1

#         length = count
#         if length != 0:
#             weight_ss = weight_ss / length
#             weight_tt = weight_tt / length
#             weight_st = weight_st / length
#         else:
#             weight_ss = np.array([0])
#             weight_tt = np.array([0])
#             weight_st = np.array([0])
#         return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')