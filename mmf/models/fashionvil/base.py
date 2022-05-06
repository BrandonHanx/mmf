# Copyright (c) Facebook, Inc. and its affiliates.

import os
from typing import Dict, List, Optional, Tuple

import torch
from mmf.modules.embeddings import BertVisioLinguisticEmbeddings
from mmf.modules.hf_layers import BertEncoderJit
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.transform import (
    transform_to_batch_sequence,
    transform_to_batch_sequence_dim,
)
from omegaconf import OmegaConf
from torch import Tensor, nn
from transformers.modeling_bert import (
    BertConfig,
    BertPooler,
    BertPreTrainedModel,
)


class FashionViLBase(BertPreTrainedModel):
    def __init__(
        self,
        config,
        visual_embedding_dim=2048,
        output_attentions=False,
        output_hidden_states=False,
    ):
        super().__init__(config)
        self.config = config

        config.visual_embedding_dim = visual_embedding_dim
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states

        self.embeddings = BertVisioLinguisticEmbeddings(config)
        self.encoder = BertEncoderJit(config)
        self.pooler = BertPooler(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.bypass_transformer = config.bypass_transformer

        self.init_weights()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        visual_embeddings: Optional[Tensor] = None,
        visual_embeddings_type: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:

        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if not torch.jit.is_scripting():
                extended_attention_mask = extended_attention_mask.to(
                    dtype=next(self.parameters()).dtype
                )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids,
            token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
        )

        encoded_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = encoded_layers[0]
        pooled_output = self.pooler(sequence_output)
        attn_data_list = []

        if not torch.jit.is_scripting():
            if self.output_attentions:
                attn_data_list = encoded_layers[1:]
        else:
            assert (
                not self.output_attentions
            ), "output_attentions not supported in script mode"

        return sequence_output, pooled_output, attn_data_list

    def get_joint_embedding(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        return self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            visual_embeddings=visual_embeddings,
            visual_embeddings_type=visual_embeddings_type,
            attention_mask=attention_mask,
        )

    def get_image_embedding(
        self,
        visual_embeddings: Tensor,
        visual_embeddings_type: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        if self.bypass_transformer:
            return self.embeddings.projection(visual_embeddings), None, None
        else:
            return self.forward(
                visual_embeddings=visual_embeddings,
                visual_embeddings_type=visual_embeddings_type,
                attention_mask=attention_mask,
            )

    def get_text_embedding(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        return self.forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


class FashionViLBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        bert_model_name = getattr(config, "bert_model_name", "bert-base-uncased")
        bert_config = BertConfig.from_dict(OmegaConf.to_container(config, resolve=True))

        self.config = config
        self.bert = FashionViLBase.from_pretrained(
            bert_model_name,
            config=bert_config,
            cache_dir=os.path.join(get_mmf_cache_dir(), "distributed_{}".format(-1)),
            visual_embedding_dim=config.visual_embedding_dim,
            output_attentions=config.output_attentions,
            output_hidden_states=config.output_hidden_states,
        )

    @staticmethod
    def flatten(
        sample_list: Dict[str, Tensor],
        to_be_flattened: List[str],
        to_be_flattened_dim: List[str],
    ) -> Dict[str, Tensor]:
        for key in to_be_flattened:
            # Make sure these keys are present or otherwise set these keys to None
            sample_list[key] = transform_to_batch_sequence(sample_list[key])
        for key in to_be_flattened_dim:
            sample_list[key] = transform_to_batch_sequence_dim(sample_list[key])
        return sample_list

    def add_custom_params(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return sample_list

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        return sample_list

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample_list = self.add_custom_params(sample_list)
        sample_list = self.flatten_for_bert(sample_list)
        sample_list = self.add_post_flatten_params(sample_list)
        return self._forward(sample_list)
