import math

import torch
from torch import optim


class RAdam(optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, warmup=0):
        """On the Variance of the Adaptive Learning Rate and Beyond https://arxiv.org/abs/1908.03265"""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, warmup=warmup)
        super(RAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self._update_param(p, group)

        return loss

    def _update_param(self, p, group):
        grad = p.grad.data

        state = self.state[p]
        if not state:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)

        state['step'] += 1

        # weight decay
        weight_decay = group['weight_decay']
        if weight_decay != 0:
            grad.add_(weight_decay, p.data)

        beta1, beta2 = group['betas']
        step = state['step']
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # update exponential moving 1st moment
        exp_avg = state['exp_avg']
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        bias_corrected_avg = exp_avg.div(bias_correction1)

        # update exponential moving 2nd moment
        exp_avg_sq = state['exp_avg_sq']
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        bias_corrected_avg_sq = exp_avg_sq.div(bias_correction2)

        # compute the maximum length of the approximated SMA
        rho_inf = 2 / (1 - beta2) - 1
        # compute the length of the approximated SMA
        rho = rho_inf - 2 * step * (beta2**step) / (bias_correction2)

        # gradual warmup
        lr = group['lr']
        warmup = group['warmup']
        if warmup > step:
            lr *= step / warmup

        if rho > 4:
            # compute the variance rectification term
            r = math.sqrt((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho))
            p.data.addcdiv_(-lr * r, bias_corrected_avg, bias_corrected_avg_sq.sqrt().add(group['eps']))
        else:
            p.data.add_(-lr, bias_corrected_avg)
