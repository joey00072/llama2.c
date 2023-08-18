import math
import torch
import torch.nn as nn

# Simple implimention of Lora
# LoRA: Low-Rank Adaptation of Large Language Models
# https://arxiv.org/abs/2106.09685


class Lora(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_alpha: int,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        device: torch.device = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        self.merged = False

        self.weight = nn.Parameter(weight)

        self.bias = None
        if bias is not None:  # to make torch.compile happy
            self.bias = nn.Parameter(bias)

        self.lora_A = nn.Parameter(torch.empty((rank, in_features), device=device))
        self.lora_B = nn.Parameter(
            torch.zeros((out_features, rank), device=device)
        )  # init with 0s

        # init A
        bound = 1 / math.sqrt(self.lora_A.size(1))
        nn.init.uniform_(self.lora_A, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_output = x @ self.weight.T
        if self.merged:
            return weight_output

        lora_output = ((x @ self.lora_A.T) @ self.lora_B.T) * self.scaling
        x = weight_output + lora_output
        if self.bias is not None:
            x = x + self.bias
        return x

    def merge(self):
        self.weight += self.lora_A.clone() @ self.lora_B.clone()
        self.merged = True

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


def get_lora(linear: nn.Linear, rank: int, lora_alpha: int = 1, device=None):
    if device is None:
        device = linear.weight.device
    return Lora(
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank,
        lora_alpha=lora_alpha,
        weight=linear.weight,
        bias=linear.bias,
        device=device,
    )


def lora_traning_setup(model: nn.Module):
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # make lora trainable
    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    return model


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 4,
    lora_alpha: int = 1,
    layers_names: list[str] = None,
    device=None,
):
    if isinstance(model, nn.Linear):
        return get_lora(model, rank, lora_alpha, device)

    if isinstance(model, (nn.Module, nn.ModuleDict)):
        for key, value in model._modules.items():
            if isinstance(value, nn.Linear):
                if layers_names is None or key in layers_names:
                    model._modules[key] = get_lora(value, rank, lora_alpha, device)
            else:
                apply_lora_to_model(value, rank, lora_alpha, layers_names, device)

    if isinstance(model, (nn.ModuleList, nn.Sequential)):
        for sub_model in model:
            if isinstance(sub_model, nn.Linear):
                if layers_names is None or key in layers_names:
                    sub_model = get_lora(sub_model, rank, lora_alpha, device)
            else:
                apply_lora_to_model(sub_model, rank, lora_alpha, layers_names, device)

    lora_traning_setup(model)
    return model
