import laplacekit as lk
import pytest
import torch
from torch import nn

from laplacekit_example import last_layers


def mlp() -> nn.Sequential:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 3))


def probed(model: nn.Module) -> lk.core.Capabilities:
    batch = lk.Batch(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))

    return lk.core.probe(lk.core.ModelFn.from_module(model), lk.Categorical(), batch, params={})


def select(subset: lk.core.Subset, model: nn.Module) -> lk.core.Params:
    chosen = subset(lk.core.ModelFn.from_module(model), probed(model))
    assert not isinstance(chosen, lk.core.Selection)

    return chosen


def test_passes_the_subset_checks():
    model = mlp()

    for n in (1, 2, 3):
        lk.testing.check_subset(last_layers(n), model, probed(model))


def test_selects_the_last_two_layers():
    model = mlp()
    params = select(last_layers(2), model)

    assert list(params) == ["2.weight", "2.bias", "4.weight", "4.bias"]
    assert params["4.weight"] is model[4].weight


def test_follows_the_forward_order():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(8, 3)  # defined first, runs last
            self.body = nn.Linear(2, 8)

        def forward(self, x):
            return self.head(self.body(x).tanh())

    assert list(select(last_layers(1), Net())) == ["head.weight", "head.bias"]


def test_keeps_the_root_modules_own_parameters():
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))
            self.linear = nn.Linear(2, 3)

        def forward(self, x):
            return self.scale * self.linear(x)

    model = Scaled()
    params = select(last_layers(2), model)

    assert list(params) == ["scale", "linear.weight", "linear.bias"]
    assert params["scale"] is model.scale


def test_too_few_layers_is_refused():
    with pytest.raises(lk.CapabilityError, match=r"last_layers\(4\).*found 3"):
        select(last_layers(4), mlp())


def test_a_function_without_a_module_is_refused():
    fn = lk.core.ModelFn(lambda params, x: x)

    with pytest.raises(lk.StructureError, match="needs a module"):
        last_layers(1)(fn, probed(mlp()))


def test_n_must_be_positive():
    with pytest.raises(ValueError, match="n >= 1"):
        last_layers(0)


def test_fits_through_the_front_door():
    model = mlp()
    x, y = torch.randn(20, 2), torch.randint(0, 3, (20,))
    post = lk.laplace(model, [(x, y)], likelihood=lk.Categorical(), subset=last_layers(2))

    assert list(post.mean) == ["2.weight", "2.bias", "4.weight", "4.bias"]


def test_is_registered():
    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-example"]
    first_line = (last_layers.__doc__ or "").splitlines()[0]

    assert info.error is None and info.supported
    assert info.summary == "The docs' example plugin: the last n layers as a subset"
    assert info.provides == (("last_layers", first_line),)
