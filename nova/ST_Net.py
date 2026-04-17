import torch
import torch.nn as nn
import torch.nn.functional as F

ST_CLAMP = 10.0


class ST_Net(nn.Module):
    """Transformer-style instant encoder with the same hidden-state contract as GAT_Net."""

    def __init__(self, input_shape, args):
        super(ST_Net, self).__init__()
        self.attention_dim = args.attention_dim

        self.input_proj = nn.Linear(input_shape, self.attention_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attention_dim,
            nhead=4,
            dim_feedforward=self.attention_dim * 2,
            dropout=args.pred_dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.rnn = nn.GRUCell(self.attention_dim, self.attention_dim)

    def forward(self, obs, hidden_state):
        # obs: [batch_size, max_vehicle_num, input_shape]
        # hidden_state: [batch_size * max_vehicle_num, attention_dim]
        batch_size, max_vehicle_num, _ = obs.shape
        obs = torch.nan_to_num(obs, nan=0.0, posinf=ST_CLAMP, neginf=-ST_CLAMP)
        hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=ST_CLAMP, neginf=-ST_CLAMP)
        hidden_state = hidden_state.clamp(min=-ST_CLAMP, max=ST_CLAMP)
        padding_mask = torch.abs(obs).sum(dim=-1) == 0
        all_padded_rows = padding_mask.all(dim=1)

        encoded = F.relu(self.input_proj(obs))
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=ST_CLAMP, neginf=-ST_CLAMP)
        encoded = encoded.clamp(min=-ST_CLAMP, max=ST_CLAMP)

        if (~all_padded_rows).any():
            valid_encoded = self.encoder(
                encoded[~all_padded_rows],
                src_key_padding_mask=padding_mask[~all_padded_rows]
            )
            encoded = encoded.clone()
            encoded[~all_padded_rows] = valid_encoded

        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=ST_CLAMP, neginf=-ST_CLAMP)
        encoded = encoded.clamp(min=-ST_CLAMP, max=ST_CLAMP)
        encoded = encoded.reshape(batch_size * max_vehicle_num, self.attention_dim)

        next_hidden = self.rnn(encoded, hidden_state)
        next_hidden = torch.nan_to_num(next_hidden, nan=0.0, posinf=ST_CLAMP, neginf=-ST_CLAMP)
        return next_hidden.clamp(min=-ST_CLAMP, max=ST_CLAMP)
