"""Model definitions for the proposed instance-adaptive fusion-depth method."""

__all__ = [
    "FusionPro",
    "FusionDepthController",
    "MultiDepthFusionNetwork",
]


def __getattr__(name):
    if name == "FusionPro":
        from .backbone import FusionPro
        return FusionPro
    if name in {"FusionDepthController", "MultiDepthFusionNetwork"}:
        from .model import FusionDepthController, MultiDepthFusionNetwork
        return {
            "FusionDepthController": FusionDepthController,
            "MultiDepthFusionNetwork": MultiDepthFusionNetwork,
        }[name]
    raise AttributeError(name)
