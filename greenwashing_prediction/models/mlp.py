import torch
import torch.nn as nn

class MLP1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
    def forward(self, x):
        return self.model(x)

def build_mlp_model(model_type="mlp1"):
    if model_type == "mlp1":
        return MLP1()
    elif model_type == "mlp2":
        return MLPWithHandmadeFeatures()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class MLPWithHandmadeFeatures(nn.Module):
    def __init__(self, handmade_dim=27):
        super().__init__()
        self.bert_proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        self.final_layers = nn.Sequential(
            nn.Linear(32 + handmade_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, bert_x, handmade_x):
        bert_out = self.bert_proj(bert_x)           # [batch_size, 32]
        x = torch.cat([bert_out, handmade_x], dim=1)  # [batch_size, 38]
        y = self.final_layers(x)                    # [batch_size, 1]
        return y