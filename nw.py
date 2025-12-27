import torch
import torch.nn as nn

from constants import TOTAL_LAYERS, TOTAL_MOVES


class ResidualBlock(nn.Module):
    """
    Стандартный ResNet-блок:
    x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN + skip -> ReLU
    """
    def __init__(self, channels: int, activation=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = activation()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x  # skip connection
        out = self.act(out)
        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero-style: conv stem + несколько residual blocks, потом shared MLP,
    затем actor/critic головы.

    Важно:
    - BatchNorm корректно работает в train() при нормальном batch size.
    - В игре/инференсе ставь model.eval(), чтобы BN использовал running stats.
    """
    def __init__(
        self,
        in_channels: int = TOTAL_LAYERS,
        n_actions: int = TOTAL_MOVES,

        # Для совместимости оставил conv_channels, но в ResNet-режиме берём только первый элемент
        conv_channels=(128,),

        num_res_blocks: int = 6,

        shared_hidden=(256,),
        actor_hidden=(256, 256),
        critic_hidden=(256, 256),

        convolution_activation_function=nn.ReLU,
        fully_connected_activation_function=nn.ReLU,
        actor_activation_function=nn.ReLU,
        critic_activation_function=nn.ReLU,
    ):
        super().__init__()

        trunk_channels = int(conv_channels[0]) if len(conv_channels) > 0 else 128
        act = convolution_activation_function

        # "Stem": Conv + BN + ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            act(),
        )

        # Residual tower
        self.res_tower = nn.Sequential(*[ResidualBlock(trunk_channels, activation=act) for _ in range(num_res_blocks)])

        self.conv = nn.Sequential(self.stem, self.res_tower)

        # Shared fully-connected
        self.flatten = nn.Flatten()
        shared_layers = []
        in_dim = trunk_channels * 8 * 8  # 8x8 chess board

        for h in shared_hidden:
            shared_layers.append(nn.Linear(in_dim, h))
            shared_layers.append(fully_connected_activation_function())
            in_dim = h

        self.shared = nn.Sequential(*shared_layers)
        self.shared_out_dim = in_dim

        # Actor head
        actor_layers = []
        in_dim = self.shared_out_dim
        for h in actor_hidden:
            actor_layers.append(nn.Linear(in_dim, h))
            actor_layers.append(actor_activation_function())
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, n_actions))
        self.actor = nn.Sequential(*actor_layers)

        # Critic head
        critic_layers = []
        in_dim = self.shared_out_dim
        for h in critic_hidden:
            critic_layers.append(nn.Linear(in_dim, h))
            critic_layers.append(critic_activation_function())
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        critic_layers.append(nn.Tanh())  # value in [-1, 1]
        self.critic = nn.Sequential(*critic_layers)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten(x)
        x = self.shared(x)
        return x

    def forward(self, x: torch.Tensor):
        feat = self._features(x)
        logits = self.actor(feat)
        values = self.critic(feat).squeeze(-1)
        return logits, values

    def greedy_action(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        logits = self.actor(feat)
        return logits.argmax(dim=-1)

    def values_only(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        values = self.critic(feat).squeeze(-1)
        return values

    # Тут не используется маска легальных ходов!!!
    def logits_only(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        logits = self.actor(feat)
        return logits

    # Это жадный ход, который используется, когда нейросеть является оппонентом
    @torch.no_grad()
    def make_move(self, x, legal_mask):
        device = next(self.parameters()).device

        single_obs = False

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        if x.dim() == 3:
            # (C, H, W) -> (1, C, H, W)
            single_obs = True
            x = x.unsqueeze(0)

        logits = self.logits_only(x)

        if not isinstance(legal_mask, torch.Tensor):
            legal_mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
        else:
            legal_mask = legal_mask.to(device)

        if legal_mask.dim() == 1:
            # (A) -> (1, A)
            legal_mask = legal_mask.unsqueeze(0)

        logits[~legal_mask] = -1e9
        actions = logits.argmax(dim=-1)

        if single_obs:
            return int(actions.item())

        return actions
