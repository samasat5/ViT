import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from embed import Embedding


class Attention(nn.Module): 

    def __init__(
        self, 
        grid_size: tuple,
        dim_embed: int, 
        num_head: int, 
        dropout_rate: float = 0.,  
        bias: bool = False, 
        locat: bool = False,
        task: str = "classif",
    ) -> None:
        
        super().__init__() # patchedpositioned(x) shape: (batch_size, num_patches+1, dim_embed) = (batch, seq_len, dim_embed)
        self.locat = locat
        self.num_head = num_head
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_head
        self.scale = math.sqrt(self.dim_head)
        self.task = task
        self.num_prefix_tokens = 1 if task == "classif" else 0
        assert dim_embed % num_head == 0

        self.q = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.v = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.k = nn.Linear(dim_embed, dim_embed, bias=bias)

        if locat:
            self.grid_size = grid_size
            self.w_var = nn.Linear(self.dim_head, 2)
            self.w_alpha = nn.Linear(self.dim_head, 1)

        self.dropout = nn.Dropout(dropout_rate) # attention dropout vs projection dropout?
        # self.register_buffer("distances", precomputed_distances)  # shape (1,1,N,N,2) ???????? au lieu de calculer distances à chaque fois

    def forward(self, x): 
        B, N, E = x.shape # (B, N, E) => la forme canonique multihead : (B, H, N, Dh)
        assert E == self.dim_embed

        # Le reshape sert à découper E en H morceaux de taille Dh(dim_head)
        value = self.v(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)
        key = self.k(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2)
        query = self.q(x).reshape(B, N, self.num_head, self.dim_head).transpose(1, 2) 

        addition = None
        if self.locat:
            """
            1. addition_2d
            2. get_var_and_alpha: 
                    - var -> la variance σ² (contrôle la largeur spatiale de la Gaussian), 
                    - alpha -> l’intensité du biais (combien la localité est favorisée)
            3. gaussian_2d_nominator
            4. pad_beginning : 
                    Elle ajoute des lignes et colonnes au début de la 
                    matrice d’attention pour gérer les tokens spéciaux 
                    (ex: CLS), pas besoin pour la segmentation.

            Simplifications:
                - get_eps(fixe 1e-6/1e-4) et get_sigmoid_fn(F.softplus) doivent être remplacés par des variantes plus simples
                - grid_size reste fixe donc pas de initial_grid_size et current_grid_size :
                  Dans notre implémentation, la résolution spatiale des tokens est fixe ; 
                  par conséquent, la grille spatiale est considérée constante et aucun 
                  ajustement multi-résolution n’est appliqué.

            Pour device: 
                Tous les tenseurs intermédiaires sont créés dynamiquement sur le 
                même dispositif que les données d’entrée, garantissant une exécution 
                cohérente sur CPU ou GPU sans gestion explicite du dispositif dans les modules.
            """
            # 1.
            eps = 1e-6

            # 2. 
            q_loc = query[:, :, self.num_prefix_tokens:, :] # (B,H,N,Dh)
            var = F.softplus(self.w_var(q_loc)) + eps # (B,H,N,2)
            var = var.unsqueeze(3) # (B,H,N,1,2)
            alpha = F.softplus(self.w_alpha(q_loc)) # autocast?? B,H,N,1

            # 3.
            x_grid, y_grid = self.grid_size
            pixels = torch.stack(
                torch.meshgrid(
                    torch.arange(x_grid, device=x.device),
                    torch.arange(y_grid, device=x.device),
                    indexing="ij",
                ),
                dim=-1
            ).to(dtype=x.dtype)

            diff = pixels.unsqueeze(0).unsqueeze(1) - pixels.unsqueeze(2).unsqueeze(3) # (n0, n1, n0, n1, 2)
            distances = -0.5 * diff.pow(2) # .sum(dim=-1) = (n0, n1, n0, n1)
            distances = distances.reshape(1, 1, x_grid * y_grid, x_grid * y_grid, 2) # B, H, N, N
            
            # 1, autocast???
            gaussian = torch.exp((distances / (var + eps)).sum(dim=-1)) # B,H,N,N
            addition = alpha * gaussian # B,H,N,N

            # 4. pad_beginning : segmentation -> rien et classif -> padding
            if self.num_prefix_tokens > 0: # classif
                addition = F.pad(
                    addition, 
                    pad=(self.num_prefix_tokens, 0, self.num_prefix_tokens, 0), 
                    mode='constant',
                ) # B,H,N+1,N+1

        x = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout.p if self.training else 0.,
            attn_mask=addition,
        ) # (B, H, N, Dh)
        attention_prob = None
        x = x.transpose(1, 2).flatten(start_dim=2)
        return x, attention_prob


class MultiHeadAttention(nn.Module):  
    def __init__(
        self, 
        grid_size: int,
        dim_embed: int, 
        num_head: int,
        dropout_rate: float, 
        bias: bool = False, 
        locat: bool = False,
        task: str = "classif",
    ) -> None:
        
        super().__init__()

        self.heads = Attention(
            grid_size, dim_embed, num_head, 
            dropout_rate, bias, locat, task,
        ) 
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(dim_embed, dim_embed) 
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, attn_prob = self.heads(x)
        x = self.out(x) 
        x = self.dropout(x)
        return x, attn_prob    