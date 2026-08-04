"""Hosted Tinker backend for SkyRL Train.

This is distinct from :mod:`skyrl.tinker`, which exposes a Tinker-compatible
API backed by SkyRL compute. This package makes SkyRL Train consume Tinker's
hosted training and sampling service.
"""

from skyrl.backends.tinker.inference import TinkerInferenceClient
from skyrl.backends.tinker.runtime import TinkerRuntime, TinkerSamplerVersion
from skyrl.backends.tinker.training_backend import TinkerPolicyDispatch

__all__ = [
    "TinkerInferenceClient",
    "TinkerPolicyDispatch",
    "TinkerRuntime",
    "TinkerSamplerVersion",
]
