import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGate(nn.Module):
    """Predict whether a structure-text query needs visual inspection."""

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.ent_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.rel_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.stat_proj = nn.Linear(2, hidden_dim, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def confidence_statistics(logits, topk):
        """Return normalized entropy and inverse margin from base predictions."""
        k = min(topk, logits.size(1))
        top_scores = torch.topk(logits.detach().float(), k=k, dim=1).values
        probs = F.softmax(top_scores, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        entropy = entropy / math.log(max(k, 2))

        if k > 1:
            scale = top_scores.std(dim=1, unbiased=False).clamp_min(1e-6)
            margin = (top_scores[:, 0] - top_scores[:, 1]) / scale
            inverse_margin = torch.exp(-margin.clamp_min(0.0))
        else:
            inverse_margin = torch.zeros_like(entropy)
        return torch.stack([entropy, inverse_margin], dim=1)

    def forward(self, ent_embed, rel_embed, base_logits, stat_topk):
        stats = self.confidence_statistics(base_logits, stat_topk)
        hidden = self.ent_proj(ent_embed) + self.rel_proj(rel_embed) + self.stat_proj(stats)
        return torch.sigmoid(self.out(hidden)).squeeze(-1), stats


class UtilityGate(nn.Module):
    """Estimate whether visual correction will improve, rather than merely change, a rank."""

    def __init__(self, embed_dim, hidden_dim, n_relations):
        super().__init__()
        self.ent_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.rel_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.stat_proj = nn.Linear(4, hidden_dim, bias=False)
        self.relation_bias = nn.Embedding(n_relations, 1)
        nn.init.zeros_(self.relation_bias.weight)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def confidence_statistics(logits, topk):
        k = min(topk, logits.size(1))
        top_scores = torch.topk(logits.detach().float(), k=k, dim=1).values
        centered = top_scores - top_scores[:, :1]
        probs = F.softmax(centered, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
        entropy = entropy / math.log(max(k, 2))
        if k > 1:
            scale = top_scores.std(dim=1, unbiased=False).clamp_min(1e-6)
            normalized_margin = (top_scores[:, 0] - top_scores[:, 1]) / scale
            inverse_margin = torch.exp(-normalized_margin.clamp_min(0.0))
            support_span = (top_scores[:, 0] - top_scores[:, -1]) / scale
        else:
            inverse_margin = torch.zeros_like(entropy)
            support_span = torch.zeros_like(entropy)
        return torch.stack(
            [entropy, inverse_margin, probs[:, 0], torch.tanh(support_span / 5.0)],
            dim=1,
        )

    def forward(self, ent_embed, rel_embed, relation_ids, base_logits, stat_topk):
        stats = self.confidence_statistics(base_logits, stat_topk)
        hidden = self.ent_proj(ent_embed) + self.rel_proj(rel_embed) + self.stat_proj(stats)
        logits = self.out(hidden).squeeze(-1)
        logits = logits + self.relation_bias(relation_ids).squeeze(-1)
        return torch.sigmoid(logits), stats


class EvidenceUtilityGate(nn.Module):
    """Predict visual utility after observing target-independent evidence shifts."""

    def __init__(self, embed_dim, hidden_dim, n_relations, stat_dim):
        super().__init__()
        self.ent_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.rel_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.stat_proj = nn.Linear(stat_dim, hidden_dim, bias=False)
        self.relation_bias = nn.Embedding(n_relations, 1)
        nn.init.zeros_(self.relation_bias.weight)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, ent_embed, rel_embed, relation_ids, evidence_stats):
        hidden = (
            self.ent_proj(ent_embed)
            + self.rel_proj(rel_embed)
            + self.stat_proj(evidence_stats)
        )
        logits = self.out(hidden).squeeze(-1)
        logits = logits + self.relation_bias(relation_ids).squeeze(-1)
        return torch.sigmoid(logits)


class RoleAdapter(nn.Module):
    """A relation-conditioned bottleneck residual for one query role."""

    def __init__(self, embed_dim, bottleneck_dim, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.relation_norm = nn.LayerNorm(embed_dim)
        self.query_down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.relation_down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, embed_dim)
        # Start exactly from the validated shared checkpoint. The adapters then
        # learn residual corrections without perturbing the first forward pass.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, shared_query, relation):
        hidden = self.query_down(self.query_norm(shared_query))
        hidden = hidden + self.relation_down(self.relation_norm(relation))
        hidden = self.dropout(F.gelu(hidden))
        return self.up(hidden)


class DeepPromptExpert(nn.Module):
    """Generate a low-rank residual prompt for the upper reasoning layers.

    The shared stem establishes role-invariant semantics first.  This expert
    then changes how those semantics are composed in the upper BERT layers,
    instead of waiting until the final graph-decoder vector.
    """

    def __init__(
        self,
        embed_dim,
        model_dim,
        bottleneck_dim,
        upper_layers,
        prompt_tokens,
        dropout,
    ):
        super().__init__()
        self.source_norm = nn.LayerNorm(embed_dim)
        self.relation_norm = nn.LayerNorm(embed_dim)
        self.source_down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.relation_down = nn.Linear(embed_dim, bottleneck_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.channel_up = nn.Linear(bottleneck_dim, model_dim)
        self.token_scale = nn.Parameter(
            torch.empty(upper_layers, prompt_tokens, 1)
        )
        # Unit token/layer scales keep the zero-output initialization while
        # avoiding a second tiny factor on the first channel-up gradients.
        nn.init.ones_(self.token_scale)
        # An unseen expert is an exact identity branch when loaded on top of an
        # existing checkpoint.  Gradients reach channel_up on its first step.
        nn.init.zeros_(self.channel_up.weight)
        nn.init.zeros_(self.channel_up.bias)

    def forward(self, source_entity, relation):
        hidden = self.source_down(self.source_norm(source_entity))
        hidden = hidden + self.relation_down(self.relation_norm(relation))
        channel = self.channel_up(self.dropout(F.gelu(hidden)))
        return channel[:, None, None, :] * self.token_scale[None, :, :, :]


class RelationRoleRouter(nn.Module):
    """Route a role adapter using a relation prior and query-level utility.

    The relation branch captures the stable prior of a directional relation.
    The query branch predicts a residual utility logit from target-independent
    evidence: source entity, Backbone confidence, and adapter update magnitude.
    """

    def __init__(
        self,
        embed_dim,
        hidden_dim,
        n_directional_relations,
        dropout,
        initial_probability,
    ):
        super().__init__()
        self.relation_norm = nn.LayerNorm(embed_dim)
        self.relation_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.role_embed = nn.Embedding(2, hidden_dim)
        self.relation_bias = nn.Embedding(n_directional_relations, 1)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.source_norm = nn.LayerNorm(embed_dim)
        self.source_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.stat_proj = nn.Linear(3, hidden_dim, bias=False)
        self.query_role_embed = nn.Embedding(2, hidden_dim)
        self.query_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # Query evidence has a deliberately asymmetric inductive bias: an
        # uncertain Backbone (high entropy / low margin) and a material Expert
        # update should never be interpreted in the opposite direction merely
        # because the training split has a different positive rate.  Clamp is
        # used instead of ReLU because PyTorch keeps a usable gradient at zero,
        # while zero initialization exactly recovers the relation-only route.
        self.query_stat_weight = nn.Parameter(torch.zeros(3))
        self.query_bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.relation_bias.weight)
        initial_probability = min(max(float(initial_probability), 1e-4), 1.0 - 1e-4)
        initial_logit = math.log(initial_probability / (1.0 - initial_probability))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, initial_logit)

        # An old pure-relation checkpoint has no trained query utility.  The
        # zero-initialized final layer and statistic weights therefore make
        # strict=False loading exactly reproduce the relation-only route until
        # this branch is trained.
        nn.init.zeros_(self.query_out[-1].weight)
        nn.init.zeros_(self.query_out[-1].bias)

    @staticmethod
    def query_statistics(shared_logits, shared_query, adapter_delta, stat_topk):
        """Return entropy, normalized top-1 margin, and relative delta norm."""
        k = min(max(int(stat_topk), 2), shared_logits.size(1))
        top_scores = torch.topk(shared_logits.detach().float(), k=k, dim=1).values
        centered = top_scores - top_scores[:, :1]
        probabilities = F.softmax(centered, dim=1)
        entropy = -(
            probabilities * torch.log(probabilities.clamp_min(1e-12))
        ).sum(dim=1)
        entropy = entropy / math.log(max(k, 2))

        if k > 1:
            score_scale = top_scores.std(dim=1, unbiased=False).clamp_min(1e-6)
            normalized_margin = (
                top_scores[:, 0] - top_scores[:, 1]
            ) / score_scale
            # Bound unusually large margins while preserving their ordering.
            normalized_margin = torch.tanh(normalized_margin / 5.0)
        else:
            normalized_margin = torch.zeros_like(entropy)

        delta_rms = adapter_delta.detach().float().pow(2).mean(dim=1).sqrt()
        query_rms = shared_query.detach().float().pow(2).mean(dim=1).sqrt().clamp_min(1e-6)
        relative_delta_norm = torch.log1p(delta_rms / query_rms)
        relative_delta_norm = torch.tanh(relative_delta_norm)
        return torch.stack(
            [entropy, normalized_margin, relative_delta_norm], dim=1
        )

    def forward(
        self,
        relation,
        relation_ids,
        roles,
        source_entity=None,
        shared_logits=None,
        shared_query=None,
        adapter_delta=None,
        stat_topk=20,
        use_query_utility=True,
        monotonic_query_utility=False,
        query_residual_scale=0.75,
        return_details=False,
    ):
        hidden = self.relation_proj(self.relation_norm(relation))
        hidden = hidden + self.role_embed(roles)
        prior_logits = self.out(hidden).squeeze(-1)
        prior_logits = prior_logits + self.relation_bias(relation_ids).squeeze(-1)

        has_query_inputs = all(
            value is not None for value in (
                source_entity, shared_logits, shared_query, adapter_delta,
            )
        )
        if use_query_utility and has_query_inputs:
            query_stats = self.query_statistics(
                shared_logits, shared_query, adapter_delta, stat_topk
            )
            query_hidden = self.source_proj(
                self.source_norm(source_entity.detach())
            )
            query_hidden = query_hidden + self.query_role_embed(roles)
            if monotonic_query_utility:
                source_gain = self.query_out(query_hidden).squeeze(-1)
                stat_weight = self.query_stat_weight.clamp_min(0.0)
                monotonic_evidence = (
                    stat_weight[0] * query_stats[:, 0]
                    - stat_weight[1] * query_stats[:, 1]
                    + stat_weight[2] * query_stats[:, 2]
                )
                # A bounded residual protects the already validated relation
                # prior. Holding source_gain fixed, derivative signs with
                # respect to entropy, margin and delta norm are +, -, +.
                raw_query_utility = (
                    source_gain + self.query_bias + monotonic_evidence
                )
                query_utility_probability = torch.sigmoid(raw_query_utility)
                query_gain_logits = float(query_residual_scale) * torch.tanh(
                    raw_query_utility
                )
            else:
                # Compatibility path for the first unrestricted query-router
                # ablation and its already trained checkpoints.
                query_hidden = query_hidden + self.stat_proj(query_stats)
                query_gain_logits = self.query_out(query_hidden).squeeze(-1)
                query_utility_probability = torch.sigmoid(query_gain_logits)
        else:
            query_gain_logits = torch.zeros_like(prior_logits)
            query_utility_probability = torch.full_like(prior_logits, 0.5)
            query_stats = relation.new_zeros(relation.size(0), 3)

        route_logits = prior_logits + query_gain_logits
        route_probability = torch.sigmoid(route_logits)
        if not return_details:
            return route_probability
        return route_probability, {
            'prior_probability': torch.sigmoid(prior_logits),
            'query_gain_logit': query_gain_logits,
            'query_utility_probability': query_utility_probability,
            'query_entropy': query_stats[:, 0],
            'query_margin': query_stats[:, 1],
            'adapter_delta_norm': query_stats[:, 2],
        }


class JointEvidenceRouter(nn.Module):
    """Refine visual invocation from role-expert and query evidence.

    The existing visual utility probability remains a calibrated prior.  A
    zero-initialized residual predicts query-specific evidence gain, making old
    checkpoints exactly backward compatible while allowing the final module to
    coordinate expert computation and visual verification.
    """

    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.source_norm = nn.LayerNorm(embed_dim)
        self.relation_norm = nn.LayerNorm(embed_dim)
        self.source_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.relation_proj = nn.Linear(embed_dim, hidden_dim, bias=False)
        # visual prior logit, expert route/prior/gain, entropy, margin, delta
        self.stat_proj = nn.Linear(7, hidden_dim, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    @staticmethod
    def _logit(probability):
        probability = probability.float()
        probability = probability.clamp(1e-5, 1.0 - 1e-5)
        return torch.log(probability) - torch.log1p(-probability)

    def forward(
        self,
        source_entity,
        relation,
        visual_prior_probability,
        expert_probability,
        expert_prior_probability,
        expert_query_gain,
        backbone_entropy,
        backbone_margin,
        expert_delta_norm,
        return_details=False,
    ):
        statistics = torch.stack(
            [
                self._logit(visual_prior_probability),
                self._logit(expert_probability),
                self._logit(expert_prior_probability),
                torch.tanh(expert_query_gain),
                backbone_entropy,
                backbone_margin,
                expert_delta_norm,
            ],
            dim=1,
        ).detach()
        # These quantities are evidence *about* whether visual inspection is
        # useful.  Stop the visual objective from changing their meaning: the
        # expert Router is supervised by rank gain, while this Router learns a
        # residual over the frozen evidence statistics.
        hidden = self.source_proj(self.source_norm(source_entity.detach()))
        hidden = hidden + self.relation_proj(self.relation_norm(relation.detach()))
        hidden = hidden + self.stat_proj(statistics)
        gain_logit = self.out(hidden).squeeze(-1)
        probability = torch.sigmoid(statistics[:, 0] + gain_logit)
        if not return_details:
            return probability
        return probability, gain_logit


class VisualVerifier(nn.Module):
    """Relation-conditioned visual matching over a small candidate set."""

    def __init__(self, embed_dim):
        super().__init__()
        self.head_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.rel_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.tail_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))

    def forward(self, head_image, relation, candidate_images):
        query = self.encode_query(head_image, relation)
        candidates = F.normalize(self.tail_proj(candidate_images), p=2, dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * torch.bmm(candidates, query.unsqueeze(-1)).squeeze(-1)

    def encode_query(self, head_image, relation):
        query = torch.tanh(self.head_proj(head_image)) * torch.tanh(self.rel_proj(relation))
        return F.normalize(query, p=2, dim=-1)

    def score_all(self, head_image, relation, entity_images):
        """Score every entity without materializing a B x N x D tensor."""
        query = self.encode_query(head_image, relation)
        candidates = F.normalize(self.tail_proj(entity_images), p=2, dim=-1)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        return scale * torch.mm(query, candidates.transpose(0, 1))
