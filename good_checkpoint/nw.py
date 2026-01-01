import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import TOTAL_LAYERS, TOTAL_MOVES


class ResidualBlock(nn.Module):
    """Standard ResNet block.

    x -> Conv(3x3) -> BN -> ReLU -> Conv(3x3) -> BN -> +skip -> ReLU
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

        out = out + x
        out = self.act(out)
        return out


class AlphaZeroNet(nn.Module):
    """AlphaZero-style CNN trunk with convolutional policy & WDL value heads.

    Changes vs your previous net:
    - No shared MLP; everything stays convolutional until the final flatten for the policy.
    - Value head predicts W/D/L logits (3 classes). We still provide a scalar value in [-1,1]
      for MCTS backup as: v = P(win) - P(loss).

    Output:
      policy_logits: (B, 4672)
      value_scalar : (B,) in [-1, 1]
      wdl_logits   : (B, 3) where classes are [WIN, DRAW, LOSS]
    """

    def __init__(
        self,
        in_channels: int = TOTAL_LAYERS,
        n_actions: int = TOTAL_MOVES,
        conv_channels=(256,),
        num_res_blocks: int = 40,
        policy_channels: int = 32,
        value_channels: int = 32,
        convolution_activation_function=nn.ReLU,
    ):
        super().__init__()

        trunk_channels = int(conv_channels[0]) if len(conv_channels) > 0 else 256
        act = convolution_activation_function

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(trunk_channels),
            act(),
        )

        # Residual tower
        self.res_tower = nn.Sequential(*[ResidualBlock(trunk_channels, activation=act) for _ in range(int(num_res_blocks))])

        self.trunk = nn.Sequential(self.stem, self.res_tower)

        # Policy head: 1x1 -> 1x1 to 73 planes (8x8x73 = 4672)
        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, int(policy_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(policy_channels)),
            act(),
            nn.Conv2d(int(policy_channels), 73, kernel_size=1, bias=True),
        )

        # Value head (WDL): 1x1 -> 1x1 to 3 planes, then global average pooling
        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, int(value_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(value_channels)),
            act(),
            nn.Conv2d(int(value_channels), 3, kernel_size=1, bias=True),
        )

        assert int(n_actions) == TOTAL_MOVES, "n_actions must match TOTAL_MOVES (8*8*73)"

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def _policy_logits(self, trunk: torch.Tensor) -> torch.Tensor:
        """Return flattened policy logits (B, 4672) with correct square orientation."""
        pol = self.policy_head(trunk)  # (B, 73, 8, 8)

        # Important: your board planes use row = 7 - rank (rank0=rank1 at bottom).
        # Move indexing uses fr = rank (0..7). To make flatten match move_to_index,
        # flip vertically so row 0 becomes rank0.
        pol = pol.flip(dims=(2,))  # (B, 73, 8, 8)

        # (B, 73, 8, 8) -> (B, 8, 8, 73) -> flatten with plane as fastest axis
        pol = pol.permute(0, 2, 3, 1).contiguous()
        return pol.view(pol.shape[0], -1)  # (B, 4672)

    def _wdl_logits(self, trunk: torch.Tensor) -> torch.Tensor:
        wdl_map = self.value_head(trunk)  # (B, 3, 8, 8)
        return wdl_map.mean(dim=(2, 3))  # (B, 3)

    @staticmethod
    def _value_from_wdl_logits(wdl_logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(wdl_logits, dim=1)
        # classes: [WIN, DRAW, LOSS]
        return probs[:, 0] - probs[:, 2]

    def forward(self, x: torch.Tensor):
        trunk = self._trunk(x)
        policy_logits = self._policy_logits(trunk)
        wdl_logits = self._wdl_logits(trunk)
        value = self._value_from_wdl_logits(wdl_logits)
        return policy_logits, value, wdl_logits

    def logits_only(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self._trunk(x)
        return self._policy_logits(trunk)

    def wdl_logits_only(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self._trunk(x)
        return self._wdl_logits(trunk)

    def values_only(self, x: torch.Tensor) -> torch.Tensor:
        trunk = self._trunk(x)
        wdl_logits = self._wdl_logits(trunk)
        return self._value_from_wdl_logits(wdl_logits)

    def greedy_action(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits_only(x)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def make_move(self, x, legal_mask):
        device = next(self.parameters()).device

        single_obs = False

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)

        if x.dim() == 3:
            single_obs = True
            x = x.unsqueeze(0)

        logits = self.logits_only(x)

        if not isinstance(legal_mask, torch.Tensor):
            legal_mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=device)
        else:
            legal_mask = legal_mask.to(device)

        if legal_mask.dim() == 1:
            legal_mask = legal_mask.unsqueeze(0)

        logits = logits.clone()
        logits[~legal_mask] = -1e9
        actions = logits.argmax(dim=-1)

        if single_obs:
            return int(actions.item())

        return actions
