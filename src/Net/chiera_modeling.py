import torch
import torch.nn as nn
from transformers import HieraPreTrainedModel, HieraConfig, HieraEmbeddings, HieraEncoder, HieraModelOutput, HieraForPreTrainingOutput
from typing import Optional, Tuple, Union

class ChessHieraConfig(HieraConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = 16  # 12 piece channels + 1 empty square + 1 color-agnostic + 1 turn + 1 influence
        self.image_size = [8, 8]  # 8x8 chess board
        self.patch_size = [1, 1]  # Each square is a patch
        self.patch_stride = [1, 1]
        self.patch_padding = [0, 0]
        self.mask_ratio = 0.25  # Default JEPA mask ratio

class ChessHieraModel(HieraPreTrainedModel):
    config_class = ChessHieraConfig

    def __init__(self, config: ChessHieraConfig):
        super().__init__(config)
        self.config = config
        
        self.embeddings = HieraEmbeddings(config, is_mae=False)
        self.encoder = HieraEncoder(config)
        
        self.init_weights()

    def create_jepa_mask(self, influence_map: torch.Tensor, mask_ratio: float = 0.25, min_region_size: int = 2, max_region_size: int = 4) -> torch.Tensor:
        batch_size, _, height, width = influence_map.shape
        num_squares = height * width
        num_masked = int(num_squares * mask_ratio)
        
        # Create initial mask based on influence
        _, indices = torch.topk(influence_map.reshape(batch_size, -1), k=num_masked, dim=1)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=influence_map.device)
        mask.view(batch_size, -1).scatter_(1, indices, False)
        
        # Expand mask regions
        for _ in range(max_region_size - 1):
            expanded_mask = nn.functional.max_pool2d(
                (~mask).float().unsqueeze(1),
                kernel_size=3, stride=1, padding=1
            ).squeeze(1).bool()
            mask = ~expanded_mask
            
            if (~mask).sum(dim=(1,2)).min() >= num_masked * max_region_size / min_region_size:
                break
        
        # Trim excess masked squares
        excess = (~mask).sum(dim=(1,2)) - num_masked
        for i in range(batch_size):
            if excess[i] > 0:
                trim_indices = torch.randperm((~mask[i]).sum())[:excess[i]]
                mask[i][~mask[i]] |= torch.zeros((~mask[i]).sum(), dtype=torch.bool, device=mask.device).scatter_(0, trim_indices, True)
        
        return mask

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HieraModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create JEPA mask based on influence channel
        influence_map = pixel_values[:, -1:, :, :]
        jepa_mask = self.create_jepa_mask(influence_map, mask_ratio=self.config.mask_ratio)

        # Apply JEPA mask to input
        masked_input = pixel_values * jepa_mask.unsqueeze(1).float()

        embedding_output = self.embeddings(masked_input, interpolate_pos_encoding=interpolate_pos_encoding)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = sequence_output.mean(dim=1)

        if not return_dict:
            return (sequence_output, pooled_output, jepa_mask) + encoder_outputs[1:]

        return HieraModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            bool_masked_pos=jepa_mask,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )

class ChessHieraForPreTraining(HieraPreTrainedModel):
    def __init__(self, config: ChessHieraConfig):
        super().__init__(config)
        self.hiera = ChessHieraModel(config)
        self.decoder = nn.Linear(config.hidden_size, config.num_channels)
        self.loss = nn.MSELoss()

    def forward(
        self,
        pixel_values: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, HieraForPreTrainingOutput]:
        outputs = self.hiera(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs.last_hidden_state
        jepa_mask = outputs.bool_masked_pos

        # Reshape sequence_output to match the input shape
        batch_size, seq_len, hidden_size = sequence_output.shape
        reconstructed_output = self.decoder(sequence_output).view(batch_size, self.config.num_channels, 8, 8)

        # Calculate loss only on masked regions
        loss = self.loss(
            reconstructed_output * (1 - jepa_mask.unsqueeze(1).float()),
            pixel_values * (1 - jepa_mask.unsqueeze(1).float())
        )

        if not return_dict:
            output = (reconstructed_output, jepa_mask) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return HieraForPreTrainingOutput(
            loss=loss,
            logits=reconstructed_output,
            bool_masked_pos=jepa_mask,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )