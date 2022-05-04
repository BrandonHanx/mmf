# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.models.fashionvil.classification import FashionViLForClassification
from mmf.models.fashionvil.composition import FashionViLForComposition
from mmf.models.fashionvil.contrastive import FashionViLForContrastive
from mmf.models.fashionvil.pretraining import FashionViLForPretraining
from mmf.utils.build import build_image_encoder
from mmf.utils.distributed import broadcast_tensor
from mmf.utils.general import filter_grads
from mmf.utils.modeling import get_fashionvil_configured_parameters
from numpy.random import choice
from torch import Tensor


@registry.register_model("fashionvil")
class FashionViL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.training_head_type = config.training_head_type

        self.double_view = config.get("double_view", False)
        self.freeze_image_encoder = config.get("freeze_image_encoder", False)
        self.freeze_transformer = config.get("freeze_transformer", False)

        if self.training_head_type == "pretraining":
            self.task_for_inference = config.task_for_inference
            self.tasks = config.tasks
            self.tasks_sample_ratio = config.get("tasks_sample_ratio", None)

    @classmethod
    def config_path(cls):
        return "configs/models/fashionvil/defaults.yaml"

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_image_encoder:
            self.image_encoder.eval()
        if self.freeze_transformer:
            self.model.bert.eval()

    def build(self):
        self.image_encoder = build_image_encoder(
            self.config.image_encoder, self.config.direct_features_input
        )
        if self.freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if self.training_head_type == "pretraining":
            self.model = FashionViLForPretraining(self.config)
        elif self.training_head_type == "classification":
            self.model = FashionViLForClassification(self.config)
        elif self.training_head_type == "composition":
            self.model = FashionViLForComposition(self.config)
        elif self.training_head_type == "contrastive":
            self.model = FashionViLForContrastive(self.config)
        else:
            raise NotImplementedError
        if self.freeze_transformer:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        weight_decay = config.optimizer.params.weight_decay
        lr_multiplier = self.config.lr_multiplier

        image_encoder_params = [
            {
                "params": filter_grads(self.image_encoder.parameters()),
                "lr": base_lr * lr_multiplier,
            }
        ]
        lr_filter = []
        lr_filter.append("bert.embeddings.projection.weight")
        lr_filter.append("bert.embeddings.projection.bias")
        if self.training_head_type == "classification":
            lr_filter.append("classifier")
        elif self.training_head_type == "pretraining":
            lr_filter.append("heads")
        bert_params = get_fashionvil_configured_parameters(
            self.model,
            base_lr,
            weight_decay,
            lr_filter,
            lr_multiplier,
        )
        return image_encoder_params + bert_params

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.training_head_type == "pretraining":
            if self.training:
                random_idx = choice(len(self.tasks), p=self.tasks_sample_ratio)
                random_idx = broadcast_tensor(torch.tensor(random_idx).cuda())
                sample_list.task = self.tasks[random_idx.item()]
            else:
                sample_list.task = self.task_for_inference

        if self.training_head_type == "composition":
            sample_list.ref_image = self.image_encoder(sample_list.ref_image)
            sample_list.tar_image = self.image_encoder(sample_list.tar_image)
        # elif self.double_view and self.training:
        #     sample_list.original_image = torch.cat(
        #         (sample_list.image, sample_list.dv_image), dim=0
        #     )
        #     sample_list.image_0 = self.image_encoder(sample_list.image)
        #     sample_list.image_1 = self.image_encoder(sample_list.dv_image)
        #     sample_list.image = torch.cat(
        #         (sample_list.image_0, sample_list.image_1), dim=0
        #     )
        else:
            if sample_list.image.dim() > 4:
                sample_list.image = torch.flatten(sample_list.image, end_dim=-4)
                sample_list.image_id = torch.flatten(sample_list.image_id, end_dim=-2)
            if sample_list.input_mask.dim() > 2:
                sample_list.input_mask = torch.flatten(
                    sample_list.input_mask, end_dim=-2
                )
                sample_list.input_ids = torch.flatten(sample_list.input_ids, end_dim=-2)
                sample_list.segment_ids = torch.flatten(
                    sample_list.segment_ids, end_dim=-2
                )
            if sample_list.targets.dim() > 1:
                sample_list.targets = torch.flatten(sample_list.targets)
            sample_list.original_image = sample_list.image
            sample_list.image = self.image_encoder(sample_list.image)

        if self.training_head_type == "pretraining" and sample_list.task == "icc":
            sample_list.dv_image = self.image_encoder(sample_list.dv_image)

        return self.model(sample_list)
