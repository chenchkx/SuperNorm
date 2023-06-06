

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from dgl.ops import segment
from utils.utils_practice import *

# Fast Matrix Power Normalized SPD Matrix Function
class FastMPNSPDMatrixFunction(Function):
    @staticmethod
    def forward(ctx, input, iterN):
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        normA = (1.0 / 3.0) * x.mul(I3).sum(dim=1).sum(dim=1)
        A = x.div(normA.view(batchSize, 1, 1).expand_as(x))
        Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad=False, device=x.device).type(dtype)
        Z = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, iterN, 1, 1).type(dtype)
        if iterN < 2:
            ZY = 0.5 * (I3 - A)
            YZY = A.bmm(ZY)
        else:
            ZY = 0.5 * (I3 - A)
            Y[:, 0, :, :] = A.bmm(ZY)
            Z[:, 0, :, :] = ZY
            for i in range(1, iterN - 1):
                ZY = 0.5 * (I3 - Z[:, i - 1, :, :].bmm(Y[:, i - 1, :, :]))
                Y[:, i, :, :] = Y[:, i - 1, :, :].bmm(ZY)
                Z[:, i, :, :] = ZY.bmm(Z[:, i - 1, :, :])
            YZY = 0.5 * Y[:, iterN - 2, :, :].bmm(I3 - Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]))
        y = YZY * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        ctx.save_for_backward(input, A, YZY, normA, Y, Z)
        ctx.iterN = iterN
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, A, ZY, normA, Y, Z = ctx.saved_tensors
        iterN = ctx.iterN
        x = input
        batchSize = x.data.shape[0]
        dim = x.data.shape[1]
        dtype = x.dtype
        der_postCom = grad_output * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
        der_postComAux = (grad_output * ZY).sum(dim=1).sum(dim=1).div(2 * torch.sqrt(normA))
        I3 = 3.0 * torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(dtype)
        if iterN < 2:
            der_NSiter = 0.5 * (der_postCom.bmm(I3 - A) - A.bmm(der_postCom))
        else:
            dldY = 0.5 * (der_postCom.bmm(I3 - Y[:, iterN - 2, :, :].bmm(Z[:, iterN - 2, :, :])) -
                          Z[:, iterN - 2, :, :].bmm(Y[:, iterN - 2, :, :]).bmm(der_postCom))
            dldZ = -0.5 * Y[:, iterN - 2, :, :].bmm(der_postCom).bmm(Y[:, iterN - 2, :, :])
            for i in range(iterN - 3, -1, -1):
                YZ = I3 - Y[:, i, :, :].bmm(Z[:, i, :, :])
                ZY = Z[:, i, :, :].bmm(Y[:, i, :, :])
                dldY_ = 0.5 * (dldY.bmm(YZ) -
                               Z[:, i, :, :].bmm(dldZ).bmm(Z[:, i, :, :]) -
                               ZY.bmm(dldY))
                dldZ_ = 0.5 * (YZ.bmm(dldZ) -
                               Y[:, i, :, :].bmm(dldY).bmm(Y[:, i, :, :]) -
                               dldZ.bmm(ZY))
                dldY = dldY_
                dldZ = dldZ_
            der_NSiter = 0.5 * (dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
        der_NSiter = der_NSiter.transpose(1, 2)
        grad_input = der_NSiter.div(normA.view(batchSize, 1, 1).expand_as(x))
        grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
        for i in range(batchSize):
            grad_input[i, :, :] += (der_postComAux[i] \
                                    - grad_aux[i] / (normA[i] * normA[i])) \
                                   * torch.ones(dim, device=x.device).diag().type(dtype)
        return grad_input, None


class DKEPooling(nn.Module):
    def __init__(self, iterN=5, snr_value=15, min_trace=1e-12, min_power=1e-12):
        super(DKEPooling, self).__init__()
        ### parameters for batch_gaussperturbation
        self.register_buffer('snr_value', torch.tensor(snr_value))
        self.register_buffer('iterN', torch.tensor(iterN))

        ### parameters for batch_gaussperturbation & batch_gaussperturbation_snr
        self.noise_weight = nn.Parameter(torch.Tensor(1))
        self.noise_bias = nn.Parameter(torch.Tensor(1))
        self.noise_weight.data.fill_(1)
        self.noise_bias.data.fill_(0)
        self.perturbation = nn.Parameter(torch.Tensor(1))
        self.perturbation.data.fill_(0)
        self.register_buffer('min_trace', torch.tensor(min_trace))
        self.register_buffer('min_power', torch.tensor(min_power))

    def batch_gaussperturbation(self, tensor, noise_rate=1e-2):
        noise = torch.randn(tensor.shape[0], tensor.shape[1]).to(tensor.device)
        return noise_rate*noise

    def batch_cov_perturbation(self, batch_cov):
        assert batch_cov.shape[1] == batch_cov.shape[2]
        if self.perturbation < 1e-5:
            self.perturbation.data.fill_(1e-5)
        batch_trace = batch_tensor_trace(batch_cov)
        batch_trace[batch_trace<self.min_trace] = self.min_trace 
        batch_noise = torch.eye(batch_cov.shape[1]).repeat(batch_cov.shape[0],1,1).to(batch_cov.device)
        return self.perturbation*batch_trace*batch_noise

    def batch_cov_perturbation_snr(self, batch_cov, snr_value):
        assert batch_cov.shape[1] == batch_cov.shape[2]
        noise = torch.randn(batch_cov.size()).to(batch_cov.device)
        batch_noise = noise.bmm(noise.transpose(1,2))/batch_cov.shape[1]
        batch_noise = self.noise_weight*batch_noise+self.noise_bias
        signal_power= batch_cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)/(batch_cov.shape[1]*batch_cov.shape[2]) # signal power
        signal_power[torch.where(signal_power<=self.min_power)] = self.min_power
        noise_power = signal_power /(10**(snr_value / 10))   # noise power
        noise_power = torch.sqrt(noise_power)/torch.std(noise, dim=[1,2])
        noise_power = noise_power.detach()
        return batch_noise*noise_power.reshape(batch_cov.shape[0],1,1)

    def forward(self, graph, feat):
        batch_nodes = graph.batch_num_nodes()
        batch_index = torch.arange(len(batch_nodes)).to(feat.device).repeat_interleave(batch_nodes) 
        feat = feat + self.batch_gaussperturbation(feat)
        batch_mean = segment.segment_reduce(batch_nodes, feat, reducer='mean') # a segment toolkit in dgl
        feat_mean = repeat_tensor_interleave(batch_mean, batch_nodes)
        feat_diff = feat - feat_mean
        batch_diff, _ = to_batch_tensor(feat_diff, batch_nodes, batch_index)
        batch_cov = batch_diff.transpose(1, 2).bmm(batch_diff)
        batch_cov = batch_cov/((batch_nodes-1).reshape(batch_nodes.shape[0],1,1))
        batch_cov = FastMPNSPDMatrixFunction.apply(batch_cov, self.iterN)

        return batch_cov.bmm(batch_mean.unsqueeze(2)).squeeze(2)

