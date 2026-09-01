import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_spatial_batches(x):
    """Convert [B, N, T, C] to [B*T, N, C] without mixing axes."""
    batch_size, num_nodes, steps, channels = x.shape
    return (
        x.permute(0, 2, 1, 3)
        .contiguous()
        .view(batch_size * steps, num_nodes, channels)
    )


def _from_spatial_batches(x, batch_size, steps):
    """Convert [B*T, N, C] back to [B, N, T, C]."""
    _, num_nodes, channels = x.shape
    return (
        x.view(batch_size, steps, num_nodes, channels)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


class StandardGCN(nn.Module):
    """Kipf-Welling GCN: sigma(D^-1/2 (A+I) D^-1/2 X W + b)."""

    def __init__(self, in_features, out_features, activation=F.relu):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

    @staticmethod
    def normalize_adjacency(adjacency):
        if adjacency is None:
            raise ValueError("Adjacency matrix is required.")
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError(
                f"Adjacency must be square, got shape {tuple(adjacency.shape)}."
            )
        adjacency = torch.nan_to_num(
            adjacency, nan=0.0, posinf=0.0, neginf=0.0
        )
        identity = torch.eye(
            adjacency.size(0), dtype=adjacency.dtype, device=adjacency.device
        )
        adjacency_hat = adjacency + identity
        degree = adjacency_hat.sum(dim=1).clamp_min(1e-12)
        degree_inv_sqrt = degree.pow(-0.5)
        return (
            degree_inv_sqrt[:, None]
            * adjacency_hat
            * degree_inv_sqrt[None, :]
        )

    def forward(self, x, normalized_adjacency):
        aggregated = torch.einsum(
            "ij,bj...->bi...", normalized_adjacency, x
        )
        output = self.linear(aggregated)
        if self.activation is not None:
            output = self.activation(output)
        return output


class CARE(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w_q = nn.Linear(in_features, out_features)
        self.w_k = nn.Linear(in_features, out_features)
        self.w_v = nn.Linear(in_features, out_features)
        self.conv_encoder = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.encoder_linear = nn.Linear(
            in_features // 2, in_features // 2
        )
        self.gru = nn.GRU(
            in_features // 2,
            in_features // 2,
            bidirectional=False,
            batch_first=True,
        )
        self.decoder_linear = nn.Linear(in_features // 2, out_features)
        self.layer_norm = nn.LayerNorm(in_features)
        self.change_bias = nn.Parameter(torch.zeros(1, 1, 1))

    def forward(self, x):
        x = self.layer_norm(x)
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        query = self.conv_encoder(query.permute(0, 2, 1))
        query = query.permute(0, 2, 1)
        query = F.leaky_relu(self.encoder_linear(query))
        query, _ = self.gru(query)
        query = F.leaky_relu(self.decoder_linear(query))

        scores = (
            torch.matmul(query, key.transpose(-2, -1))
            + self.change_bias
        ) / (key.size(-1) ** 0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), value)


class GM(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.glu_linear = nn.Linear(in_dim, mid_dim * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(mid_dim)
        self.linear2 = nn.Linear(mid_dim, out_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        y = self.norm1(x)
        y = F.glu(self.glu_linear(y), dim=-1)
        y = self.dropout1(F.gelu(y))
        z = self.dropout2(F.gelu(self.linear2(self.norm2(y))))
        return x + z if x.shape[-1] == z.shape[-1] else z


class CrossAttention(nn.Module):
    def __init__(self, query_features, output_features):
        super().__init__()
        self.w_q = nn.Linear(query_features, output_features)
        self.w_k = nn.Linear(query_features, output_features)
        self.w_v = nn.Linear(query_features, output_features)

    def forward(self, query_input, key_input, value_input):
        query = self.w_q(query_input)
        key = self.w_k(key_input)
        value = self.w_v(value_input)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (key.size(-1) ** 0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), value)


class SelfAttention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.w_q = nn.Linear(features, features)
        self.w_k = nn.Linear(features, features)
        self.w_v = nn.Linear(features, features)

    def forward(self, x):
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (key.size(-1) ** 0.5)
        return torch.matmul(torch.softmax(scores, dim=-1), value)


class C2F_RCM(nn.Module):
    """Coarse-to-Fine Regional Context Modeling (C2F-RCM)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.care = CARE(hidden_dim, hidden_dim)
        self.coarse_gcn = StandardGCN(hidden_dim, hidden_dim)
        self.fine_gcn = StandardGCN(hidden_dim, hidden_dim)

    def forward(self, x_base, coarse_adj, fine_adj):
        batch_size, _, steps, _ = x_base.shape
        context = self.care(_to_spatial_batches(x_base))
        context = _from_spatial_batches(context, batch_size, steps)
        x_global = self.coarse_gcn(context, coarse_adj)
        x_local = self.fine_gcn(context, fine_adj)
        return x_global, x_local


class G_FGM(nn.Module):
    """Graph-based Fine-grained Feature Generation Module (G-FGM)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.feature_generator = GM(
            hidden_dim, 4 * hidden_dim, 2 * hidden_dim, dropout=0.1
        )
        self.fluctuation_gcn = StandardGCN(
            2 * hidden_dim, 2 * hidden_dim
        )
        self.local_residual = nn.Linear(hidden_dim, 2 * hidden_dim)

    def forward(self, x_base, x_local, fluctuation_adj):
        batch_size, _, steps, _ = x_base.shape
        generated = self.feature_generator(_to_spatial_batches(x_base))
        generated = self.fluctuation_gcn(generated, fluctuation_adj)
        generated = _from_spatial_batches(generated, batch_size, steps)
        residual = F.leaky_relu(self.local_residual(x_local))
        return generated + residual


class G_CGM(nn.Module):
    """Graph-based Coarse-grained Feature Generation Module (G-CGM)."""

    def __init__(self, hidden_dim):
        super().__init__()
        coarse_dim = hidden_dim // 2
        self.feature_generator = GM(
            hidden_dim, 2 * hidden_dim, coarse_dim, dropout=0.1
        )
        self.dcor_gcn = StandardGCN(coarse_dim, coarse_dim)
        self.granger_gcn = StandardGCN(coarse_dim, coarse_dim)
        self.global_residual = nn.Linear(hidden_dim, coarse_dim)

    def forward(self, x_base, x_global, dcor_adj, granger_adj):
        batch_size, _, steps, _ = x_base.shape
        coarse = self.feature_generator(_to_spatial_batches(x_base))
        coarse = _from_spatial_batches(coarse, batch_size, steps)
        x_dcor = self.dcor_gcn(coarse, dcor_adj)
        x_granger = self.granger_gcn(coarse, granger_adj)
        residual = F.leaky_relu(self.global_residual(x_global))
        return x_dcor * x_granger + residual


class TS_DGFM(nn.Module):
    """Trend-Seasonality Dynamic Gated Fusion Module (TS-DGFM)."""

    def __init__(self, hidden_dim, dd_branch, gate_mode="adaptive"):
        super().__init__()
        if gate_mode not in {"adaptive", "static_global", "no_gate"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.dd_branch = dd_branch
        self.dd_projection = nn.Linear(1, hidden_dim)
        self.seasonal_gcn = StandardGCN(hidden_dim, hidden_dim)
        self.trend_gcn = StandardGCN(hidden_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dropout=0.1,
            batch_first=True,
        )
        self.seasonal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )
        self.trend_bigru = nn.GRU(
            hidden_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.trend_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.gate = nn.Linear(3 * hidden_dim, hidden_dim)
        self.static_gate = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(
        self,
        data,
        x_base,
        seasonal_adj,
        trend_adj,
        device,
        ddgcrn_i,
        return_aux=False,
    ):
        batch_size, num_nodes, steps, _ = x_base.shape
        y_dd = self.dd_branch(data, device=device, i=ddgcrn_i)
        y_dd_features = self.dd_projection(
            y_dd.permute(0, 2, 1, 3).contiguous()
        )
        y_dd_features = y_dd_features.view(
            batch_size * num_nodes, steps, -1
        )

        seasonal = self.seasonal_gcn(x_base, seasonal_adj)
        seasonal = seasonal.view(batch_size * num_nodes, steps, -1)
        seasonal = self.seasonal_encoder(seasonal)

        trend = self.trend_gcn(x_base, trend_adj)
        trend = trend.view(batch_size * num_nodes, steps, -1)
        trend, _ = self.trend_bigru(trend)
        trend = self.trend_projection(trend)

        if self.gate_mode == "adaptive":
            gate = torch.sigmoid(
                self.gate(
                    torch.cat([y_dd_features, seasonal, trend], dim=-1)
                )
            )
        elif self.gate_mode == "static_global":
            gate = torch.sigmoid(self.static_gate).expand_as(trend)
        else:
            gate = torch.full_like(trend, 0.5)
        fused = trend * gate + seasonal * (1.0 - gate)
        fused = fused.view(batch_size, num_nodes, steps, -1)
        if return_aux:
            return fused, y_dd, {
                "ts_dgfm_gate": gate.view(batch_size, num_nodes, steps, -1),
                "ts_dgfm_seasonal": seasonal.view(
                    batch_size, num_nodes, steps, -1
                ),
                "ts_dgfm_trend": trend.view(
                    batch_size, num_nodes, steps, -1
                ),
            }
        return fused, y_dd


class DS_STF(nn.Module):
    """Dual-Scale Spatial-Temporal Fusion (DS-STF)."""

    def __init__(
        self,
        hidden_dim,
        output_steps,
        kernel_size=3,
        mode="cross_attention",
    ):
        super().__init__()
        if mode not in {
            "cross_attention",
            "concat_projection",
            "no_cross_alignment",
        }:
            raise ValueError(f"Unsupported DS_STF mode: {mode}")
        self.mode = mode
        self.base_to_fine = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.base_to_coarse = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fine_attention = CrossAttention(
            2 * hidden_dim, hidden_dim
        )
        self.coarse_attention = CrossAttention(
            hidden_dim // 2, hidden_dim
        )
        self.fine_concat_projection = nn.Linear(4 * hidden_dim, hidden_dim)
        self.coarse_concat_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fine_direct_projection = nn.Linear(2 * hidden_dim, hidden_dim)
        self.coarse_direct_projection = nn.Linear(hidden_dim // 2, hidden_dim)
        self.spatial_conv = nn.Conv1d(
            in_channels=3 * hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.self_attention = SelfAttention(hidden_dim)
        self.temporal_projection = nn.Linear(
            hidden_dim, 2 * hidden_dim
        )
        self.temporal_conv = nn.Conv1d(
            in_channels=2 * hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.output_bigru = nn.GRU(
            2 * hidden_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            2 * hidden_dim, output_steps
        )

    def forward(self, x_base, x_fine, x_coarse, x_temporal):
        batch_size, num_nodes, steps, _ = x_base.shape
        fine_query = self.base_to_fine(x_base)
        coarse_query = self.base_to_coarse(x_base)
        if self.mode == "cross_attention":
            fine = self.fine_attention(fine_query, x_fine, x_fine)
            coarse = self.coarse_attention(
                coarse_query, x_coarse, x_coarse
            )
        elif self.mode == "concat_projection":
            fine = self.fine_concat_projection(
                torch.cat([fine_query, x_fine], dim=-1)
            )
            coarse = self.coarse_concat_projection(
                torch.cat([coarse_query, x_coarse], dim=-1)
            )
        else:
            fine = self.fine_direct_projection(x_fine)
            coarse = self.coarse_direct_projection(x_coarse)

        stacked = torch.cat(
            [
                _to_spatial_batches(x_temporal),
                _to_spatial_batches(fine),
                _to_spatial_batches(coarse),
            ],
            dim=-1,
        )
        spatial = self.spatial_conv(stacked.permute(0, 2, 1))
        spatial = F.leaky_relu(spatial).permute(0, 2, 1)
        spatial = _from_spatial_batches(spatial, batch_size, steps)

        attended = self.self_attention(spatial)
        temporal = self.temporal_projection(attended)
        temporal = temporal.view(batch_size * num_nodes, steps, -1)
        temporal = self.temporal_conv(temporal.permute(0, 2, 1))
        temporal = F.leaky_relu(temporal).permute(0, 2, 1)
        temporal, _ = self.output_bigru(temporal)
        output = temporal[:, -1, :].view(batch_size, num_nodes, -1)
        output = self.output_projection(output)
        return output.permute(0, 2, 1).contiguous().unsqueeze(-1)


class MGCSTEPBackbone(nn.Module):
    """Production MGC-STEP backbone assembled from paper-named modules."""

    GRAPH_NAMES = (
        "A_Base_const",
        "A_Flu_const",
        "A_Coarse_Modular_const",
        "A_Fine_Modular_const",
        "A_Dcor_const",
        "A_Granger_const",
        "A_DTW_Seasonal_const",
        "A_DTW_Trend_const",
    )

    def __init__(
        self,
        hidden_dim,
        output_steps,
        dd_branch,
        kernel_size=3,
        disabled_modules=None,
        ds_stf_mode="cross_attention",
        gate_mode="adaptive",
    ):
        super().__init__()
        self.disabled_modules = set(disabled_modules or [])
        self.output_steps = output_steps
        self.base_gcn = StandardGCN(1, hidden_dim)
        self.c2f_rcm = C2F_RCM(hidden_dim)
        self.g_fgm = G_FGM(hidden_dim)
        self.g_cgm = G_CGM(hidden_dim)
        self.ts_dgfm = TS_DGFM(hidden_dim, dd_branch, gate_mode=gate_mode)
        self.ds_stf = DS_STF(
            hidden_dim, output_steps, kernel_size, mode=ds_stf_mode
        )
        self.fine_bypass = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.coarse_bypass = nn.Linear(hidden_dim, hidden_dim // 2)
        self.simple_output_projection = nn.Linear(hidden_dim, output_steps)
        for graph_name in self.GRAPH_NAMES:
            self.register_buffer(graph_name, None)

    def _simple_output(self, x_temporal):
        output = self.simple_output_projection(x_temporal[:, :, -1, :])
        return output.permute(0, 2, 1).contiguous().unsqueeze(-1)

    def forward(
        self,
        data,
        device,
        database_tag="GENERIC",
        ddgcrn_i=2,
        return_aux=False,
    ):
        del database_tag
        missing = [
            name for name in self.GRAPH_NAMES
            if getattr(self, name) is None
        ]
        if missing:
            raise RuntimeError(
                "Graphs must be injected before forward: "
                + ", ".join(missing)
            )

        flow_x = data["flow_x"].to(device)
        if flow_x.dim() == 3:
            flow_x = flow_x.unsqueeze(0)

        aux = {}
        x_base = self.base_gcn(flow_x, self.A_Base_const)
        if "C2F_RCM" in self.disabled_modules:
            x_global, x_local = x_base, x_base
        else:
            coarse_adj = self.A_Coarse_Modular_const
            fine_adj = self.A_Fine_Modular_const
            if "single_community" in self.disabled_modules:
                fine_adj = coarse_adj
            x_global, x_local = self.c2f_rcm(
                x_base, coarse_adj, fine_adj
            )

        if "G_FGM" in self.disabled_modules or "coarse_only" in self.disabled_modules:
            x_fine = self.fine_bypass(x_base)
        else:
            x_fine = self.g_fgm(x_base, x_local, self.A_Flu_const)

        if "G_CGM" in self.disabled_modules or "fine_only" in self.disabled_modules:
            x_coarse = self.coarse_bypass(x_base)
        else:
            x_coarse = self.g_cgm(
                x_base,
                x_global,
                self.A_Dcor_const,
                self.A_Granger_const,
            )

        if "TS_DGFM" in self.disabled_modules:
            x_temporal = x_base
            y_dd = torch.zeros(
                x_base.size(0),
                self.output_steps,
                x_base.size(1),
                1,
                dtype=x_base.dtype,
                device=x_base.device,
            )
        else:
            temporal_result = self.ts_dgfm(
                data,
                x_base,
                self.A_DTW_Seasonal_const,
                self.A_DTW_Trend_const,
                device,
                ddgcrn_i,
                return_aux=return_aux,
            )
            if return_aux:
                x_temporal, y_dd, temporal_aux = temporal_result
                aux.update(temporal_aux)
            else:
                x_temporal, y_dd = temporal_result

        if "DS_STF" in self.disabled_modules:
            output = self._simple_output(x_temporal)
        else:
            output = self.ds_stf(
                x_base, x_fine, x_coarse, x_temporal
            )
        output = output + y_dd

        if return_aux:
            aux.update(
                {
                    "x_base": x_base,
                    "x_global": x_global,
                    "x_local": x_local,
                    "x_fine": x_fine,
                    "x_coarse": x_coarse,
                    "x_temporal": x_temporal,
                }
            )
            return output, aux
        return output
