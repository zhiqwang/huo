"""PyTorch modules and reconstruction; install the torch extra to use them."""

try:
    from huo.torch.projector import FanbeamProjector
    from huo.torch.reconstruction import reconstruct
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise ModuleNotFoundError(
            "The PyTorch backend requires the 'torch' extra: uv sync --extra torch", name="torch"
        ) from error
    raise

__all__ = ["FanbeamProjector", "reconstruct"]
