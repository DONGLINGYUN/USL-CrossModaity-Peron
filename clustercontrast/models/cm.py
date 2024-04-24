import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd

class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):

        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))

class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    # @staticmethod
    # def forward(ctx, inputs, targets, features, momentum):
    #     ctx.features = features
    #     ctx.momentum = momentum
    #     ctx.save_for_backward(inputs, targets)
    #     outputs = inputs.mm(ctx.features.t())
    #
    #     return outputs
    #
    # @staticmethod
    # def backward(ctx, grad_outputs):
    #     inputs, targets = ctx.saved_tensors
    #     grad_inputs = None
    #     if ctx.needs_input_grad[0]:
    #         grad_inputs = grad_outputs.mm(ctx.features)
    #
    #     batch_centers = collections.defaultdict(list)
    #     for instance_feature, index in zip(inputs, targets.tolist()):
    #         batch_centers[index].append(instance_feature)
    #
    #     for index, features in batch_centers.items():
    #         distances = []
    #         for feature in features:
    #             distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
    #             distances.append(distance.cpu().numpy())
    #
    #         median = np.argmin(np.array(distances))
    #         ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
    #         ctx.features[index] /= ctx.features[index].norm()
    #
    #     return grad_inputs, None, None, None
    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features) // 2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index + nums] = ctx.features[index + nums] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index + nums] /= ctx.features[index + nums].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.cross_entropy = nn.CrossEntropyLoss().cuda()
        self.register_buffer('features', torch.zeros(2*num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.use_hard == True:
            # outputs = cm_hard(inputs, targets, self.features, self.momentum)
            outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            output_hard, output_mean = torch.chunk(outputs, 2, dim=1)
            ##
            # out = torch.stack(output_mean, dim=0)
            # neg = torch.max(output_mean, dim=0)[0]
            # pos = torch.min(output_mean, dim=0)[0]
            # mask = torch.zeros_like(output_mean[0]).scatter_(1, targets.unsqueeze(1), 1)
            # logits = mask * pos + (1 - mask) * neg
            ##

            loss = 0.5 *(F.cross_entropy(output_hard, targets) + 0.5* F.cross_entropy(output_mean, targets))
            # loss = 0.5 * (self.cross_entropy(output_hard, targets) + (1 - 0.5) * self.cross_entropy (output_mean, targets))
            return loss
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)

            outputs /= self.temp
            loss = F.cross_entropy(outputs, targets)
            return loss
