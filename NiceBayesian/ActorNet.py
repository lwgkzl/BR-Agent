import torch
import numpy as np
from torch import nn
from typing import Any, Dict, Tuple, Union, Optional, Sequence
from tianshou.data import to_torch
from tianshou.utils.net.common import Net
import copy
# from NewBayesNetwork.untils import logging


class MyActor(nn.Module):

    def __init__(
        self,
        args,
        preprocess_net: nn.Module,
        action_shape: Sequence[int],
        transformer_matrix: np.array = None,
        transformer_matrix2: np.array = None,
        prob_threshold: float = 0.95,
        # 0 表示只使用relation_matrix， 1表示只使用mutual_info  2表示只使用mlp  3表示3个综合  4表示用mlp来学习一个参数来调节这两个矩阵
        # feature_model: int = 0
    ) -> None:
        super().__init__()
        # print("our actor")
        self.action_shape = action_shape
        self.preprocess = preprocess_net
        self.feature_model = args.feature_model
        tt = transformer_matrix
        if self.feature_model == 1:
            tt = transformer_matrix2
        if self.feature_model == 3:
            if transformer_matrix2 is None:
                print("error, need transformer_matrix2 in actornet.py")
            tt += transformer_matrix2
        if self.feature_model == 4:
            if args.transfor_grad:
                self.transformer_matrix2 = nn.Parameter(torch.tensor(transformer_matrix2, dtype=torch.float32))
            else:
                self.transformer_matrix2 = nn.Parameter(torch.tensor(transformer_matrix2, dtype=torch.float32), requires_grad=False)
            # self.transformer_matrix2 = nn.Parameter(torch.tensor(transformer_matrix2, dtype=torch.float32), requires_grad=False)

        if args.transfor_grad:
            self.transformer_matrix = nn.Parameter(torch.tensor(tt, dtype=torch.float32))
        else:
            self.transformer_matrix = nn.Parameter(torch.tensor(tt, dtype=torch.float32), requires_grad=False)
        self.device = args.device
        self.max_turn = args.max_episode_steps
        self.threshold = prob_threshold
        self.disease_num = self.transformer_matrix.size(0)
        self.symptom_num = self.transformer_matrix.size(-1)
        self.mlp_net = None

        if self.feature_model >= 2:
            if self.feature_model == 4:
                self.mlp_net = Net(layer_num=2, state_shape=tuple([self.disease_num + self.symptom_num, ]),
                                   action_shape=1, softmax=False)  # learn a beita
            else:
                self.mlp_net = Net(layer_num=2, state_shape=tuple([self.disease_num+self.symptom_num, ]), action_shape=self.symptom_num, softmax=True)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        s = to_torch(s,  device=self.device, dtype=torch.float32)
        disease_probs = self.preprocess(s)  # bs * disease_num
        disease_probs = torch.clamp(disease_probs, 0.0, 1.0)
        aa = disease_probs
        if self.feature_model <= 1:  # 只使用transformatrix
            logits = torch.max(aa.unsqueeze(2) * self.transformer_matrix, 1)[0]  # bs * symptom_num
        elif self.feature_model == 2:  # 只使用MLP
            mlp_input = torch.cat((aa, s[:, self.disease_num:]), -1)
            logits = self.mlp_net(mlp_input)[0]
        elif self.feature_model == 3:  # 三个相加
            trans_logits = torch.max(aa.unsqueeze(2) * self.transformer_matrix, 1)[0]
            mlp_input = torch.cat((aa, s[:, self.disease_num:]), -1)
            mlp_logits = self.mlp_net(mlp_input)[0]
            logits = trans_logits + mlp_logits
        else:
            mlp_input = torch.cat((aa, s[:, self.disease_num:]), -1)
            beita = torch.sigmoid(self.mlp_net(mlp_input)[0])
            beita = beita.unsqueeze(1)
            trans = self.transformer_matrix * beita + (torch.ones_like(beita) - beita) * self.transformer_matrix2
            logits = torch.max(aa.unsqueeze(2) * trans, 1)[0]  # bs * symptom_num

        if 'history' in info.keys():
            history = torch.tensor(info['history'][:, self.disease_num:])
            mask = torch.ones_like(history) - history
        else:
            mask = torch.ones_like(logits)

        logits = logits * mask

        disease_num = disease_probs.size(-1)
        logits = torch.cat((disease_probs.float(), logits.float()), 1)
        for k in range(disease_probs.size(0)):
            if (torch.max(disease_probs[k]) >= self.threshold and not self.training) or mask[k].sum() == 0 or logits[k][self.disease_num:].sum()==0. or info.get('turn', np.zeros(disease_probs.size(0)))[k] >= self.max_turn:
                logits[k][disease_num:] = torch.zeros(self.symptom_num)
            else:
                logits[k][:disease_num] = torch.zeros(disease_num)

        logits = to_torch(logits,  device=self.device, dtype=torch.float32)
        return logits, state
