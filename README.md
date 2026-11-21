# laplacekit-example

A template for <code><b>laplacekit</b></code> plugins. It ships one subset,
`last_layers(n)`, which puts the posterior on the last `n` layers of a model:

```python
import laplacekit as lk
from laplacekit_example import last_layers

post = lk.laplace(model, data, likelihood=lk.Categorical(), subset=last_layers(2))
```

The docs page [Your first plugin](https://laplacekit.com/plugins/first.html)
explains the code.

## Layout

```
pyproject.toml                    # package metadata, entry point, tool settings
src/laplacekit_example/
    __init__.py                   # last_layers and the plugin record
    py.typed                      # marks the package as typed
tests/test_last_layers.py         # the tests
```

`lk.plugins()` finds the package through its entry point in `pyproject.toml`:

```toml
[project.entry-points."laplacekit.plugins"]
laplacekit_example = "laplacekit_example:plugin"
```

`plugin` is the `lk.Plugin` record at the bottom of `__init__.py`.

## Start your own

1. Click **Use this template** on GitHub, or copy the files.
2. Rename the package: `laplacekit-example` becomes `laplacekit-<name>` and
   `laplacekit_example` becomes `laplacekit_<name>`, in `pyproject.toml`, the
   `src/` folder and the tests.
3. Replace `last_layers` with your own code and tests.

## Develop

```bash
uv sync                # the package, laplacekit and the dev tools
uv run pytest          # tests
uv run ruff check      # lint
uv run ty check        # types
uv run lint-imports    # only the public laplacekit API is imported
```

> [!NOTE]
> Until <code><b>laplacekit</b></code> is on PyPI, the `dev` group in
> `pyproject.toml` installs it from GitHub. Delete that line once it is.

## License

MIT
