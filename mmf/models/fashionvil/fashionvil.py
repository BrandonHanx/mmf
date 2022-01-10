# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

from mmf.common.registry import registry
from mmf.models import BaseModel
from mmf.models.fashionvil.classification import FashionViLForClassification
from mmf.models.fashionvil.composition import FashionViLForComposition
from mmf.models.fashionvil.contrastive import FashionViLForContrastive
from mmf.models.fashionvil.pretraining import FashionViLForPretraining
from mmf.utils.build import build_image_encoder
from mmf.utils.general import filter_grads
from torch import Tensor


@registry.register_model("fashionvil")
class FashionViL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.training_head_type = config.training_head_type

    @classmethod
    def config_path(cls):
        return "configs/models/fashionvil/defaults.yaml"

    def build(self):
        self.image_encoder = build_image_encoder(
            self.config.image_encoder, self.config.direct_features_input
        )
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

        if self.config.special_visual_initialize:
            self.model.bert.embeddings.initialize_visual_from_pretrained()

        if getattr(self.config, "freeze_base", False):
            for p in self.model.bert.parameters():
                p.requires_grad = False

    def get_optimizer_parameters(self, config):
        base_lr = config.optimizer.params.lr
        image_encoder_params = [
            {
                "params": filter_grads(self.image_encoder.parameters()),
                "lr": base_lr * 20,
            }
        ]
        rest_params = [
            {"params": filter_grads(self.model.parameters()), "lr": base_lr},
        ]
        training_parameters = image_encoder_params + rest_params

        return training_parameters

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.training_head_type == "composition":
            sample_list.ref_image = self.image_encoder(sample_list.ref_image)
            sample_list.tar_image = self.image_encoder(sample_list.tar_image)
        else:
            sample_list.image = self.image_encoder(sample_list.image)
        return self.model(sample_list)
