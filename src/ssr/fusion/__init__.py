"""Fusion utilities."""

from ssr.fusion.attention import AttentionFusionFit, fit_attention_fusion
from ssr.fusion.project import FusionFit, collect_user_blocks, fit_fusion
from ssr.fusion.registry import fit_fusion_method

__all__ = [
    "AttentionFusionFit",
    "FusionFit",
    "collect_user_blocks",
    "fit_attention_fusion",
    "fit_fusion",
    "fit_fusion_method",
]
