from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm 
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)

        self.A = torch.nn.Parameter(torch.zeros(in_features, lora_dim, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(lora_dim, out_features, dtype=torch.float32))
        
        # torch.nn.init.normal_(self.A, mean=0.5, std=0.0000002)
        # torch.nn.init.normal_(self.B, mean=0.0, std=0.0000002)
        torch.nn.init.normal_(self.A)
        torch.nn.init.normal_(self.B)
        
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float32 = x.to(torch.float32)
        output = super().forward(x_float32)
        lora_output = torch.matmul(x_float32, self.A).matmul(self.B)
        
        return output + lora_output


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
