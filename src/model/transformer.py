import torch
import torch.nn as nn


class GeneTransformer(nn.Module):
    def __init__(self, n_genes, d=192, n_layers=4, n_heads=4, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d
        self.n_genes = n_genes
        self.d = d
        self.gene_emb = nn.Embedding(n_genes, d)
        self.value_proj = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
                                           dim_feedforward=d_ff, dropout=dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        scale = d ** -0.5
        nn.init.normal_(self.gene_emb.weight, std=scale)
        nn.init.normal_(self.value_proj[0].weight, std=scale)
        nn.init.zeros_(self.value_proj[0].bias)
        nn.init.normal_(self.value_proj[2].weight, std=scale)
        nn.init.zeros_(self.value_proj[2].bias)

    def embed(self, values, mask):
        b, g = values.shape
        ids = torch.arange(g, device=values.device).expand(b, g)
        h = self.gene_emb(ids) + self.value_proj(values.unsqueeze(-1))
        return torch.where(mask.unsqueeze(-1),
                           self.gene_emb(ids) + self.mask_emb, h)

    def forward(self, values, mask, return_hidden=False):
        h = self.encoder(self.embed(values, mask))
        out = self.head(h).squeeze(-1)
        return (out, h) if return_hidden else out

    def layer_states(self, values, mask):
        """Приховані стани ПІСЛЯ кожного шару, включно з вкладенням (індекс 0)."""
        h = self.embed(values, mask)
        out = [h]
        for layer in self.encoder.layers:
            h = layer(h)
            out.append(h)
        if self.encoder.norm is not None:
            out[-1] = self.encoder.norm(out[-1])
        return out

    def run_patched(self, values, mask, patches=None):
        """patches: список (шар, позиція, тензор) — підміна прихованого стану
        ПІСЛЯ вказаного шару. Шар 0 = одразу після вкладення."""
        patches = patches or []
        by_layer = {}
        for li, pos, vec in patches:
            by_layer.setdefault(li, []).append((pos, vec))
        h = self.embed(values, mask)
        for pos, vec in by_layer.get(0, []):
            h = h.clone()
            h[:, pos, :] = vec
        for i, layer in enumerate(self.encoder.layers, start=1):
            h = layer(h)
            for pos, vec in by_layer.get(i, []):
                h = h.clone()
                h[:, pos, :] = vec
        if self.encoder.norm is not None:
            h = self.encoder.norm(h)
        return self.head(h).squeeze(-1)

    def attention_maps(self, values, mask, per_head=False):
        h = self.embed(values, mask)
        maps = []
        for layer in self.encoder.layers:
            x = layer.norm1(h)
            _, w = layer.self_attn(x, x, x, need_weights=True,
                                   average_attn_weights=not per_head)
            maps.append(w)
            h = layer(h)
        return torch.stack(maps, 1)
