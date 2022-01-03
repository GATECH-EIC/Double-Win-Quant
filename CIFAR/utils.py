import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
std = torch.tensor(cifar10_std).view(3,1,1).cuda()

upper_limit = ((1 - mu)/ std)
lower_limit = ((0 - mu)/ std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)



def get_loaders(dir_, batch_size, dataset='cifar10', norm=True):
    if norm:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        dataset_normalization = None

    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(cifar10_mean, cifar10_std),
        ])
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=cifar10_mean, std=cifar10_std)

    num_workers = 2
    
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100(
            dir_, train=False, transform=test_transform, download=True)
    else:
        print('Wrong dataset:', dataset)
        exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )
    return train_loader, test_loader, dataset_normalization


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts, opt=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            if opt is not None:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X+delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


def evaluate_pgd(test_loader, model, attack_iters, restarts, logger, args, log_detail=True):
    epsilon = (args.epsilon / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()

        model.module.set_precision(num_bits=args.num_bits, num_grad_bits=0)

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

        model.module.set_precision(num_bits=args.num_bits, num_grad_bits=0)

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        if i % 10 == 0 and log_detail:
                logger.info("Iter: [{:d}/{:d}]\t"
                             "Prec@1 {:.3f} ({:.3f})\t".format(
                                i,
                                len(test_loader),
                                (output.max(1)[1] == y).sum().item() / y.size(0),
                                pgd_acc/n)
                )

    return pgd_loss/n, pgd_acc/n


def evaluate_pgd_random(test_loader, model, attack_iters, restarts, logger, args, num_bits_list=None, num_round=10, log_detail=True):
    epsilon = (args.epsilon / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    if num_bits_list is None:
        assert args.num_bits_list is not None
        num_bits_list = args.num_bits_list

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            model.module.set_precision(num_bits=np.random.choice(num_bits_list), num_grad_bits=0)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

            model.module.set_precision(num_bits=np.random.choice(num_bits_list), num_grad_bits=0)

            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

            if i % 10 == 0 and log_detail:
                logger.info("Iter: [{:d}][{:d}/{:d}]\t"
                            "Prec@1 {:.3f} ({:.3f})\t".format(
                                r,
                                i,
                                len(test_loader),
                                (output.max(1)[1] == y).sum().item() / y.size(0),
                                pgd_acc/n)
                )

    return pgd_loss/n, pgd_acc/n


def evaluate_pgd_heter_prec(test_loader, model, attack_iters, restarts, logger, args, log_detail=True):
    epsilon = (args.epsilon / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    assert args.num_attack_bits is not None

    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()

        model.module.set_precision(num_bits=args.num_attack_bits, num_grad_bits=0)

        pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

        model.module.set_precision(num_bits=args.num_bits, num_grad_bits=0)

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            pgd_loss += loss.item() * y.size(0)
            pgd_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)

        if i % 10 == 0 and log_detail:
            logger.info("Iter: [{:d}/{:d}]\t"
                        "Prec@1 {:.3f} ({:.3f})\t".format(
                            i,
                            len(test_loader),
                            (output.max(1)[1] == y).sum().item() / y.size(0),
                            pgd_acc/n)
            )

    return pgd_loss/n, pgd_acc/n


def evaluate_pgd_heter_prec_random(test_loader, model, attack_iters, restarts, logger, args, num_round=1, log_detail=True):
    epsilon = (args.epsilon / 255.) / std
    alpha = (2 / 255.) / std
    pgd_loss = 0
    pgd_acc = 0
    n = 0
    model.eval()

    for r in range(num_round):
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()

            model.module.set_precision(num_bits=np.random.choice(args.num_attack_bits_list), num_grad_bits=0)

            pgd_delta = attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts)

            model.module.set_precision(num_bits=np.random.choice(args.num_bits_list), num_grad_bits=0)

            with torch.no_grad():
                output = model(X + pgd_delta)
                loss = F.cross_entropy(output, y)
                pgd_loss += loss.item() * y.size(0)
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)

            if i % 10 == 0 and log_detail:
                logger.info("Iter: [{:d}/{:d}]\t"
                            "Prec@1 {:.3f} ({:.3f})\t".format(
                                i,
                                len(test_loader),
                                (output.max(1)[1] == y).sum().item() / y.size(0),
                                pgd_acc/n)
                )

    return pgd_loss/n, pgd_acc/n



def evaluate_standard(test_loader, model, args):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    model.module.set_precision(num_bits=args.num_bits, num_grad_bits=0)

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n



def evaluate_standard_random(test_loader, model, args, num_bits_list=None):
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    if num_bits_list is None:
        assert args.num_bits_list is not None
        num_bits_list = args.num_bits_list

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            model.module.set_precision(num_bits=np.random.choice(num_bits_list), num_grad_bits=0)

            X, y = X.cuda(), y.cuda()
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n



def set_random_precision(args, _iters, logger):
    num_bits_range = np.arange(args.num_bits_schedule[0], args.num_bits_schedule[1]+1)
    args.num_bits = np.random.choice(num_bits_range)

    if _iters % 10 == 0:
        logger.info('Random Precision: Iter [{}] num_bits = {}'.format(_iters, args.num_bits))