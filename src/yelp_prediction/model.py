import torch
import torch.nn as nn


class MILModel(nn.Module):
    def __init__(
        self,
        *,
        init_rating=4.0,
        input_dim=1280,
        attention_dim=256,
    ):
        super().__init__()


        # Gated Attention
        self.attention_v = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(input_dim, attention_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attention_dim, 1)


        # Simple Regression Head
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        # careful: this is technically incorrect: we're using the median of the
        # whole dataset, not just our split
        #initial_bias = torch.logit(torch.tensor((median_stars - 1) / 4))
        #nn.init.constant_(self.head[-1].bias, initial_bias)
        # correcting this nto using an initial bias based on the average of the training set 
        init_prob = (init_rating - 1) / 4
        init_prob = torch.clamp(torch.tensor(init_prob), 1e-4, 1 - 1e-4)
        init_bias = torch.logit(init_prob)

        nn.init.constant_(self.head[-1].bias, init_bias)

    def forward(self, x):
       # x shape: [batch, num_photos, input_dim]

        # 1. Calcolo i due rami dell'attenzione
        v = self.attention_v(x) # [batch, photos, attention_dim]
        u = self.attention_u(x) # [batch, photos, attention_dim]
    
        # 2. Gated mechanism: moltiplicazione elemento per elemento
        # Questo permette al modello di "chiudere" il cancello sulle foto inutili
        gated_attention = v * u 
    
        # 3. Trasformo in punteggi (scores)
        scores = self.attention_weights(gated_attention) # [batch, photos, 1]

        # --- AGGIUNTA FONDAMENTALE: MASKING PER IL PADDING ---
        # Creiamo una maschera che è True dove i vettori sono tutti zero (padding)
        # Sommiamo i valori assoluti delle feature: se è 0, è padding.
        mask = (x.abs().sum(dim=-1) == 0).unsqueeze(-1) 
        # Mettiamo un valore molto basso (-inf) dove c'è padding per azzerare il Softmax
        scores = scores.masked_fill(mask, -1e9) 
        # -----------------------------------------------------

        # 4. Softmax per ottenere i pesi (somma = 1)
        weights = torch.softmax(scores, dim=1) # [batch, photos, 1]

        # 5. Aggregazione (Bag Feature)
        bag_feature = (weights * x).sum(dim=1) # [batch, input_dim]

        # 6. Regressione finale
        raw_output = self.head(bag_feature)
        return torch.sigmoid(raw_output) * 4 + 1

class SinglePhotoModel(nn.Module):
    """
    Model for single-photo prediction (non-MIL approach).

    Unlike MILModel which aggregates multiple photos with mean pooling,
    this model directly predicts from a single photo feature vector.
    """

    def __init__(
        self,
        *,
        median_stars=4.0,
        input_dim=1280,
    ):
        super().__init__()

        # Same regression head as MILModel
        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

        # Initialize bias based on median (same as MIL approach)
        initial_bias = torch.logit(torch.tensor((median_stars - 1) / 4))
        nn.init.constant_(self.head[-1].bias, initial_bias)

    def forward(self, x):
        """
        Forward pass for single photo.

        Args:
            x: Feature tensor of shape [B, FEATURE_DIM]
               (NOT [B, K, FEATURE_DIM] like in MIL)

        Returns:
            Predictions of shape [B, 1] in range [1, 5]
        """
        # No mean pooling needed - already single photo!
        # Just pass through the regression head
        raw_output = self.head(x)

        # Same sigmoid scaling as MIL model
        # Sigmoid -> [0, 1] * 4 -> [0, 4] + 1 -> [1, 5]
        return torch.sigmoid(raw_output) * 4 + 1
