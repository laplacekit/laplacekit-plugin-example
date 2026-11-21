"""The example plugin from the docs, a parameter subset of the last `n` layers."""

import laplacekit as lk
from laplacekit.core import (
    ROOT_BLOCK,
    Capabilities,
    CapabilityError,
    ModelFn,
    Params,
    StructureError,
    Subset,
)

__all__ = ["last_layers"]


def last_layers(n: int) -> Subset:
    """Select the parameters of the last `n` modules with parameters that ran in the probe.

    Args:
        n: The number of modules, at least 1.

    Returns:
        A `Subset`. Called with the model and the probe's capabilities, it returns the
        parameters that those modules hold themselves, by their full names. A call
        raises `StructureError` when the model has no module, and `CapabilityError` when
        the probe found fewer than `n` modules with parameters.

    Raises:
        ValueError: `n` is below 1.
    """

    if n < 1:
        raise ValueError(f"last_layers needs n >= 1, got {n}")

    def subset(model: ModelFn, caps: Capabilities, /) -> Params:
        if model.module is None:
            raise StructureError("last_layers needs a module, but ModelFn.module is None")

        found = caps.module_order

        if len(found) < n:
            raise CapabilityError(
                f"last_layers({n}): the probe found {len(found)} parameterized modules"
            )

        params = {}

        for name in found[-n:]:
            module = model.module if name == ROOT_BLOCK else model.module.get_submodule(name)
            prefix = "" if name == ROOT_BLOCK else f"{name}."

            for own, p in module.named_parameters(recurse=False):
                params[prefix + own] = p

        return params

    return subset


plugin = lk.Plugin(supports=">=0.1a0,<0.2")
"""The record behind the `laplacekit.plugins` entry point."""
