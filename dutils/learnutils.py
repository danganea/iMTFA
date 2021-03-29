import torch
from torch.nn import Parameter


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def count_trainable_named_parameters(model: torch.nn.Module):
    return {n: p.numel() for (n, p) in model.named_parameters() if p.requires_grad}


def count_frozen_named_parameters(model: torch.nn.Module):
    return {n: p.numel() for (n, p) in model.named_parameters() if not p.requires_grad}


def get_trainable_named_parameters(model: torch.nn.Module):
    return [n for (n, p) in model.named_parameters() if p.requires_grad]


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])


def freeze_all_parameters(submodule: torch.nn.Module) -> None:
    for p in submodule.parameters():
        p.requires_grad_(False)
    submodule.eval()
