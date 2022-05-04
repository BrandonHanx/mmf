# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
import torch.nn as nn
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from torch import Tensor


class FashionViLForComposition(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.norm_layer = NormalizationLayer()
        self.enable_prompt = config.get("enable_prompt", False)
        if self.enable_prompt:
            self.prefix_prompts = nn.Parameter(torch.zeros(1, 10, 768))
        else:
            self.prefix_prompts = None

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["ref_image"].shape
        device = sample_list["ref_image"].device

        sample_list["tar_visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        sample_list["ref_visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        if self.prefix_prompts is None:
            sample_list["comp_attention_mask"] = torch.cat(
                (sample_list["input_mask"], torch.ones((b, l), device=device).long()),
                dim=-1,
            )
        else:
            sample_list["comp_attention_mask"] = torch.cat(
                (
                    torch.ones((b, self.prefix_prompts.size(1)), device=device).long(),
                    sample_list["input_mask"],
                    torch.ones((b, l), device=device).long(),
                ),
                dim=-1,
            )
        if not self.config.bypass_transformer and self.prefix_prompts is not None:
            sample_list["visual_attention_mask"] = torch.ones(
                (b, self.prefix_prompts.size(1) + l), device=device
            ).long()
        else:
            sample_list["visual_attention_mask"] = torch.ones(
                (b, l), device=device
            ).long()
        return sample_list

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids", "segment_ids"]
        to_be_flattened_dim = ["ref_image", "tar_image"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        tar_embeddings, _, _ = self.bert.get_image_embedding(
            sample_list["tar_image"],
            sample_list["tar_visual_embeddings_type"],
            sample_list["visual_attention_mask"],
            self.prefix_prompts if not self.config.bypass_transformer else None,
        )
        tar_embeddings = tar_embeddings.mean(dim=1)
        tar_embeddings = self.norm_layer(tar_embeddings)

        comp_embeddings, _, _ = self.bert.get_joint_embedding(
            sample_list["input_ids"],
            sample_list["segment_ids"],
            sample_list["ref_image"],
            sample_list["ref_visual_embeddings_type"],
            sample_list["comp_attention_mask"],
            self.prefix_prompts,
        )
        num_visual_tokens = sample_list["tar_image"].shape[1]
        comp_embeddings = comp_embeddings[:, -num_visual_tokens:].mean(dim=1)
        comp_embeddings = self.norm_layer(comp_embeddings)

        output_dict = {
            "scores": comp_embeddings,
            "targets": tar_embeddings,
        }
        return output_dict
