# Copyright 2025 EPFL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Based on 4M, BEiT, timm, DINO, DeiT code bases
# https://github.com/apple/ml-4m/
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------

import torch
from torch import optim as optim

from nanofm.modeling.muon import Muon


def create_optimizers(args, model):
    muon_params = [p for name, p in model.named_parameters() if p.ndim >= 2 and not "emb" in name]
    adamw_params = [p for name, p in model.named_parameters() if p.ndim < 2 or "emb" in name]

    adamw_opt_args = dict(lr=args.lr)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        adamw_opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        adamw_opt_args['betas'] = args.opt_betas

    muon_opt_args = dict(lr=args.lr)
    if hasattr(args, 'momentum') and args.momentum is not None:
        muon_opt_args['momentum'] = args.momentum
    if hasattr(args, 'global_rank') and args.global_rank is not None:
        muon_opt_args['rank'] = args.global_rank
    if hasattr(args, 'world_size') and args.world_size is not None:
        muon_opt_args['world_size'] = args.world_size

    print("AdamW optimizer settings:", adamw_opt_args)
    print("Muon optimizer settings:", muon_opt_args)

    optimizers = [
        Muon(muon_params, **muon_opt_args),
        optim.AdamW(adamw_params, **adamw_opt_args),
    ]

    return optimizers
