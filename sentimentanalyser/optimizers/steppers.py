from .utils import debias


def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p


def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr*wd)
    return p


weight_decay._defaults = dict(wd=0.)


def l2_reg(p, lr, wd, **kwargs):
    p.grad.data.add_(wd, p.data)
    return p


l2_reg._defaults = dict(wd=0.)


def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):
    debias1 = debias(mom, mom_damp, step)
    debias2 = debias(sqr_mom, sqr_damp, step)

    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg / debias2).sqrt() + eps)
    return p


adam_step._defaults = dict(eps=1e-5)


def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)
    return p
