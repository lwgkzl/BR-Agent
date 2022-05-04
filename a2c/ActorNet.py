import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional, Sequence
# from tianshou.data import to_torch
from tianshou.data import to_torch
def is_nan(ten):
    aa = torch.isnan(ten)
    if torch.sum(aa) >0:
        return True
    else:
        return False

def is_negative(ten):
    aa = torch.tensor(ten<0).long()
    if torch.sum(aa) >0:
        return True
    else:
        return False

class MyActor(nn.Module):
    """Simple actor network with MLP.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        hidden_layer_size: int = 128,
        softmax_output: bool = True,
        disease_num: int=5,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.last = nn.Linear(hidden_layer_size, np.prod(action_shape))
        self.softmax_output = softmax_output
        self.disease_num = disease_num
        self.least_ask =4
        self.device='cpu'
    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        s = to_torch(s, device=self.device, dtype=torch.float32)
        logits, h = self.preprocess(s, state)

        logits = self.last(logits)
        bs, action_num = s.size()

        mask = torch.ones_like(logits)

        if 'turn' in info.keys():
            for i in range(bs):
                if info['turn'][i] < self.least_ask and np.sum(info['history'][i][self.disease_num:]) < (action_num - self.disease_num):
                    mask[i][:self.disease_num] = 0
            if 'history' in info.keys():
                mask = mask * torch.tensor(np.ones_like(info['history']) - info['history'])

        if self.softmax_output:
            logits = torch.where(mask==0, torch.full_like(logits, -1e16), logits)
            logits = F.softmax(logits, dim=-1)

        return logits, h