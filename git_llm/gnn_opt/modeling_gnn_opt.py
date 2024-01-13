"""PyTorch GNN OPT model."""
import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as nnF
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, global_max_pool, global_mean_pool
from transformers import (
    OPTConfig,
    OPTForCausalLM,
    OPTModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
)
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding


def global_min_pool(
    x: torch.Tensor, batch: Optional[torch.Tensor], size: Optional[int] = None
) -> torch.Tensor:
    x = global_max_pool(-1 * x, batch, size)
    return -1 * x


class HeteroGraphFeatureExtractor(nn.Module):
    def __init__(
        self, num_blocks: int, num_classes: int, device: torch.device, hidden_channels
    ) -> None:
        super(HeteroGraphFeatureExtractor, self).__init__()
        self.num_blocks = num_blocks  # call "conv + batchnorm + leaky_relu" as a block
        self.conv_layers = []
        self.batchnorm_layers = []
        for _ in range(self.num_blocks):
            self.conv_layers.append(
                HeteroConv(
                    convs={
                        ("host", "sends", "flow"): SAGEConv(
                            hidden_channels, hidden_channels
                        ),
                        ("flow", "rev_sends", "host"): SAGEConv(
                            hidden_channels, hidden_channels
                        ),
                        ("flow", "precedes", "flow"): SAGEConv(
                            hidden_channels, hidden_channels
                        ),
                        ("flow", "reaches", "host"): SAGEConv(
                            hidden_channels, hidden_channels
                        ),
                    },
                    aggr="mean",
                ).to(device=device)  # TODO: configure aggregation method
            )
            self.batchnorm_layers.append(
                torch.nn.BatchNorm1d(hidden_channels).to(device=device)
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for block_idx in range(self.num_blocks):
            x_dict = self.conv_layers[block_idx](
                x_dict=x_dict, edge_index_dict=edge_index_dict
            )
            x_dict = {k: self.batchnorm_layers[block_idx](v) for k, v in x_dict.items()}
            x_dict = {k: nnF.leaky_relu(v) for k, v in x_dict.items()}
        return x_dict


class GNNModel(nn.Module):
    def __init__(
        self,
        num_conv_blocks: int,
        num_flow_feat: int,
        num_classes: int,
        device: torch.device,
        hidden_channels: int = 64,
    ) -> None:
        super(GNNModel, self).__init__()
        self.device = device
        self.hidden_channels = hidden_channels
        self.flow_feat_compressor = torch.nn.Linear(num_flow_feat, hidden_channels)
        self.gnn = HeteroGraphFeatureExtractor(
            num_blocks=num_conv_blocks,
            num_classes=num_classes,
            device=device,
            hidden_channels=hidden_channels,
        )
        self.readout = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data: HeteroData) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        x_dict = {
            "flow": self.flow_feat_compressor(data["flow"].x),
            "host": torch.ones(
                size=(len(data["host"].node_id), self.hidden_channels)
            ).to(
                device=self.device
            ),  # TODO: move tensor instantiation out of the function (DI)
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        x_flow_mapped_to_class = self.readout(x_dict["flow"])
        return x_dict, x_flow_mapped_to_class


class GNNEncoder(nn.Module):
    def __init__(self, gnn_model: GNNModel) -> None:
        super(GNNEncoder, self).__init__()
        self.gnn_model = gnn_model

    def forward(self, data: Batch) -> torch.Tensor:
        x_dict, _ = self.gnn_model(data)
        flow_pooled_mean = global_mean_pool(x_dict["flow"], batch=data["flow"]["batch"])
        flow_pooled_max = global_max_pool(x_dict["flow"], batch=data["flow"]["batch"])
        flow_pooled_min = global_min_pool(x_dict["flow"], batch=data["flow"]["batch"])
        host_pooled_mean = global_mean_pool(x_dict["host"], batch=data["host"]["batch"])
        host_pooled_max = global_max_pool(x_dict["host"], batch=data["host"]["batch"])
        host_pooled_min = global_min_pool(x_dict["host"], batch=data["host"]["batch"])

        return torch.concat(
            tensors=[
                flow_pooled_mean,
                flow_pooled_max,
                flow_pooled_min,
                host_pooled_mean,
                host_pooled_max,
                host_pooled_min,
            ],
            dim=1,
        )


class GNNProjection(nn.Module):
    def __init__(self, input_vec_length: int, hidden_vec_length: int):
        super(GNNProjection, self).__init__()
        self.graph_projection = nn.Sequential(
            nn.Linear(input_vec_length, hidden_vec_length),
            nn.LayerNorm(
                hidden_vec_length, eps=0.00001
            ),  # TODO: seek optimal value of eps (currently default value)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.graph_projection(embeddings)


class GNNOPTConfig(OPTConfig):
    model_type = "gnn_opt"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["model_type"] = self.__class__.model_type
        return output


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


class GNNOPTModel(OPTModel):
    config_class = GNNOPTConfig

    def __init__(self, config: OPTConfig):
        super(GNNOPTModel, self).__init__(config)

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # load my GNN
        gnn_model = GNNModel(
            num_conv_blocks=2, num_flow_feat=77, num_classes=8, device=device
        ).to(device=device)
        result = gnn_model.load_state_dict(torch.load("/tmp/gnn_weight.pth"))
        print(f"GNN Model loading result: {result}")
        gnn_model.eval()
        self.graph_encoder = GNNEncoder(gnn_model=gnn_model)

        self.num_graph_tokens = 1
        self.graph_projection = GNNProjection(
            # IMPORTANT: see the output of GNNEncoder. Length of pooled flow vectors and pooled host vectors are 64 each, so 64*6 if all vectors are concatenated.
            input_vec_length=64 * 6,
            hidden_vec_length=2560,  # IMPORTANT: This was determined by seeing OPTConfig
            # opt-350m: 1024, opt-1.3b: 2048, opt-2.7b: 2560
        )

        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _generate_future_mask(
        self, size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=dtype), diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def create_attention_mask(
        self,
        tgt,
        memory,
        tgt_mask,
        past_key_values_length,
        memory_key_padding_mask=None,
    ):
        num_tgt = tgt.shape[1]
        num_memory = memory.shape[1]
        device = tgt.device
        dtype = tgt.dtype
        top_left = torch.zeros((num_memory, num_memory), device=device, dtype=dtype)
        top_right = torch.full(
            (num_memory, num_tgt + past_key_values_length),
            float("-inf"),
            device=tgt.device,
            dtype=dtype,
        )
        bottom_left = torch.zeros(
            (num_tgt, num_memory),
            dtype=dtype,
            device=tgt_mask.device,
        )

        if past_key_values_length > 0:
            tgt_mask = torch.zeros(
                (tgt_mask.shape[0], tgt_mask.shape[0] + past_key_values_length),
                dtype=dtype,
                device=tgt_mask.device,
            )

        left = torch.cat((top_left, bottom_left), dim=0)
        right = torch.cat((top_right, tgt_mask.to(dtype)), dim=0)

        full_attention_mask = torch.cat((left, right), dim=1)[None, :]

        if memory_key_padding_mask is None:
            memory_key_padding_mask = torch.full(
                (memory.shape[0], memory.shape[1]), fill_value=False, device=device
            )
        # if it is False, it means valid. That is, it is not a padding
        if memory_key_padding_mask.dtype != torch.bool:
            raise ValueError("Memory key padding mask must be a boolean tensor.")
        zero_negative_infinity = torch.zeros_like(
            memory_key_padding_mask, dtype=tgt.dtype
        )
        zero_negative_infinity[memory_key_padding_mask] = float("-inf")
        full_attention_mask = full_attention_mask.expand(
            (
                memory_key_padding_mask.shape[0],
                num_memory + num_tgt,
                num_memory + past_key_values_length + num_tgt,
            )
        )
        full_attention_mask = full_attention_mask.clone()
        origin_left = full_attention_mask[:, :, :num_memory]
        update = zero_negative_infinity[:, None, :]
        full_attention_mask[:, :, :num_memory] = origin_left + update

        # add axis for multi-head
        full_attention_mask = full_attention_mask[:, None, :, :]

        return full_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        graph_data: Optional[HeteroData] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPooling]:
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_hidden_states
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        projected_graph_features = None
        if graph_data is not None and past_key_values is None:
            # IMPORTANT: # 1. Extract graph features using GNN
            # here we assume graph_data is a HeteroData
            graph_features = self.graph_encoder(graph_data)

            # IMPORTANT: 2. Convert features extracted by ViT into prompt-like Image Embeddings
            projected_graph_features = self.graph_projection(graph_features)
            projected_graph_features = projected_graph_features  # NOTE: temporally removed .squeeze().unsqueeze(1)

        # IMPORTANT: 3. Vectorize the tokens
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L634-L658
        inputs_embeds = self.decoder.embed_tokens(input_ids)

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length
        if past_key_values is not None:
            mask_seq_length = mask_seq_length - self.num_graph_tokens
            past_key_values_length = past_key_values_length - self.num_graph_tokens

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        # IMPORTANT: 4. Obtain Positional Encoding
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.decoder.project_in is not None:
            # NOTE: 5. Dimension adjustment of Text Embeddings specific to OPT
            inputs_embeds = self.decoder.project_in(inputs_embeds)

        # IMPORTANT: 6. Text Embeddings + Positional Encoding
        embedding_output = inputs_embeds + pos_embeds

        if projected_graph_features is None:
            # NOTE: if no image is provided, zero-fill graph feature vector
            projected_graph_features = torch.zeros(
                # NOTE:                     v Why 0?
                (embedding_output.shape[0], 0, embedding_output.shape[2]),
                dtype=embedding_output.dtype,
                device=embedding_output.device,
            )

        # Repeat graph features to match embedding batch size.
        # ( My thought: embedding can be batched. So here graph feature is repeated to be same length vector as text embedding? )
        projected_graph_features = projected_graph_features.repeat(
            embedding_output.size(0) // projected_graph_features.size(0), 1, 1
        )

        # IMPORTANT: 7. Concatenate Image Embeddings and Text Embeddings
        # concatenate patch token and text token embeddings
        hidden_states = torch.cat((projected_graph_features, embedding_output), dim=1)

        # IMPORTANT: 8. Create Causal Attention Mask for Text region
        # By default, an additive causal mask is created
        # for masking the future (one direction).
        tgt_mask = self._generate_future_mask(
            seq_length, embedding_output.dtype, embedding_output.device
        )

        # for full sequence (w/ graph patch tokens)
        if past_key_values is not None:
            past_key_values_length = past_key_values_length + self.num_graph_tokens

        # IMPORTANT: 9. Create Attention Mask for Custom Causal Attention
        # Create an attention mask of shape (batch_size, 1, tgt_seq_len, src_seq_len)
        combined_attention_mask = self.create_attention_mask(
            tgt=embedding_output,
            memory=projected_graph_features,
            tgt_mask=tgt_mask,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is not None:
            # if the user provides an attention mask, we add it to the default one
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, embedding_output.dtype, tgt_len=input_shape[-1]
            ).to(embedding_output.device)
            if past_key_values_length > 0:
                expanded_attn_mask = expanded_attn_mask[
                    :, :, -past_key_values_length:, :
                ]
            else:
                combined_attention_mask[
                    :, :, -input_shape[1] :, -input_shape[1] :
                ] += expanded_attn_mask

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # IMPORTANT: 10. Pass through the Decoder layer repeatedly, the main part of the language model
        for idx, decoder_layer in enumerate(self.decoder.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # dropout_probability = random.uniform(0, 1)
            # if self.training and (dropout_probability < self.layerdrop):
            #     continue

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.decoder.gradient_checkpointing and self.decoder.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    combined_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.decoder.final_layer_norm is not None:
            hidden_states = self.decoder.final_layer_norm(hidden_states)

        if self.decoder.project_out is not None:
            # NOTE: 11. Dimension adjustment MLP specific to OPT
            hidden_states = self.decoder.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        # IMPORTANT: 12. Align the output interface
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class GNNOPTForCausalLM(OPTForCausalLM):
    config_class = GNNOPTConfig

    def __init__(
        self,
        config,
    ):
        super(GNNOPTForCausalLM, self).__init__(config)
        self.model = GNNOPTModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        graph_data: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if labels is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            graph_data=graph_data,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            num_graph_tokens = self.model.num_graph_tokens
            shifted_logits = logits[:, num_graph_tokens:-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shifted_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "graph_data": kwargs.get("graph_data", None),
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        return reordered_past
