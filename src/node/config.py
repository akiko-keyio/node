"""Configuration management for default arguments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import itertools
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from .core import Node

__all__ = ["Config"]


class Config:
    """Store default arguments for tasks using OmegaConf.

    Configuration values may reference other nodes using ``${...}`` syntax. When
    such references are encountered, :class:`Config` will lazily build the
    referenced node with the provided :class:`Runtime` instance.
    """

    def __init__(
        self,
        mapping: Mapping[str, dict[str, Any]] | DictConfig | str | Path | None = None,
    ) -> None:
        """Create a configuration mapping.

        Parameters
        ----------
        mapping:
            Initial configuration data.
        """
        if isinstance(mapping, (str, Path)):
            try:
                object.__setattr__(self, "_conf", OmegaConf.load(str(mapping)))
            except Exception as e:
                from .exceptions import ConfigurationError
                raise ConfigurationError(f"Failed to load config from {mapping}: {e}") from e
        else:
            object.__setattr__(self, "_conf", OmegaConf.create(mapping or {}))

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self._conf[name]
        except (KeyError, TypeError):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Allow attribute-style setting of config values."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._conf[name] = value

    def _resolve_with_presets(self, cfg: DictConfig) -> DictConfig:
        """Resolve a config that may contain _presets_.
        
        If the config has a ``_presets_`` key, merge the base config
        (all keys except ``_use_`` and ``_presets_``) with the selected
        preset from ``_presets_[_use_]``.
        
        Parameters
        ----------
        cfg : DictConfig
            The configuration to resolve.
            
        Returns
        -------
        DictConfig
            The merged configuration, or the original if no presets.
        """
        if not OmegaConf.is_dict(cfg) or "_presets_" not in cfg:
            return cfg  # No presets, return as-is
        
        # Extract base config (exclude special keys)
        base = {k: v for k, v in cfg.items() if k not in ("_use_", "_presets_")}
        
        # Get selected preset name
        preset_name = cfg.get("_use_")
        if preset_name is None:
            return OmegaConf.create(base)  # No selection, return base only
        
        # Get preset config
        presets = cfg.get("_presets_", {})
        preset = presets.get(preset_name)
        if preset is None:
            return OmegaConf.create(base)  # Preset not found, return base only
        
        # Merge: base + preset (preset overrides base)
        preset_dict = OmegaConf.to_container(preset, resolve=False) if OmegaConf.is_dict(preset) else {}
        merged = {**base, **preset_dict}
        return OmegaConf.create(merged)

    def _locate(self, path: str) -> Callable[..., Any]:
        mod_name, attr = path.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[attr])
        return getattr(mod, attr)

    def _resolve_value(self, val: Any, runtime: Any) -> Any:
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            key = val[2:-1]
            if key in self._conf:
                cfg_val = self._conf[key]
                if OmegaConf.is_dict(cfg_val) and "_target_" in cfg_val:
                    return self._build_node(key, runtime)
            return OmegaConf.select(self._conf, key)
        return val

    def _normalize_sweep_axis_values(self, values: Any) -> list[Any]:
        """Normalize one sweep axis value collection into a concrete list."""
        if isinstance(values, (str, bytes)):
            return [values]
        try:
            axis_values = list(values)
        except TypeError:
            axis_values = [values]
        if not axis_values:
            raise ValueError("sweep axis values cannot be empty")
        return axis_values

    def _set_dotted_param(self, params: dict[str, Any], path: str, value: Any) -> None:
        """Set a value into params by dotted path (e.g. ``a.b.c``)."""
        keys = [k for k in path.split(".") if k]
        if not keys:
            raise ValueError("sweep path cannot be empty")
        if len(keys) == 1:
            params[keys[0]] = value
            return

        current: Any = params
        for key in keys[:-1]:
            if not isinstance(current, dict):
                raise ValueError(
                    f"sweep path '{path}' expects mapping at '{key}', got {type(current).__name__}"
                )
            if key not in current:
                raise ValueError(f"sweep path '{path}' has missing key '{key}' in params")
            current = current[key]

        if not isinstance(current, dict):
            raise ValueError(
                f"sweep path '{path}' expects mapping before leaf, got {type(current).__name__}"
            )
        current[keys[-1]] = value

    def _instantiate_with_sweep(
        self,
        *,
        name: str,
        fn: Callable[..., Any],
        params: dict[str, Any],
        sweep: Mapping[str, Any],
    ) -> "Node":
        """Instantiate a sweep root node and reuse existing DAG/dimension semantics."""
        from .core import Node
        from .exceptions import ConfigurationError
        from .hashing import _canonical

        axis_specs: list[tuple[str, list[Any], str]] = []
        for raw_path, raw_values in sweep.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ConfigurationError(f"Invalid sweep key: {raw_path!r}. Expected non-empty string path.")
            path = raw_path.strip()
            try:
                axis_values = self._normalize_sweep_axis_values(raw_values)
            except ValueError as e:
                raise ConfigurationError(f"Invalid sweep values for '{path}': {e}") from e
            dim_name = f"sweep_{path.replace('.', '_')}"
            axis_specs.append((path, axis_values, dim_name))

        if not axis_specs:
            return fn(**params)

        sweep_shape = tuple(len(values) for _, values, _ in axis_specs)
        item_nodes: list[Node] = []
        for combo in itertools.product(*(values for _, values, _ in axis_specs)):
            combo_params = deepcopy(params)
            for (path, _, _), value in zip(axis_specs, combo, strict=True):
                try:
                    self._set_dotted_param(combo_params, path, value)
                except ValueError as e:
                    raise ConfigurationError(str(e)) from e
            item_nodes.append(fn(**combo_params))

        if not item_nodes:
            raise ConfigurationError("Sweep produced no instantiated nodes.")

        sweep_dims = tuple(dim_name for _, _, dim_name in axis_specs)
        sweep_coords = {dim_name: values for _, values, dim_name in axis_specs}

        first = item_nodes[0]
        if first.dims and first._items is not None:
            child_dims = first.dims
            child_coords = first.coords
            child_shape = first._items.shape
            for node in item_nodes[1:]:
                if node.dims != child_dims or node._items is None or node._items.shape != child_shape:
                    raise ConfigurationError(
                        "All sweep instantiations must produce nodes with identical dimension layout."
                    )
                for dim in child_dims:
                    if node.coords.get(dim) != child_coords.get(dim):
                        raise ConfigurationError(
                            f"Sweep child dimension '{dim}' has inconsistent coords across combinations."
                        )

            combined_shape = sweep_shape + child_shape
            combined_items = np.empty(combined_shape, dtype=object)
            for flat_idx, node in enumerate(item_nodes):
                sweep_idx = np.unravel_index(flat_idx, sweep_shape)
                for inner_idx in np.ndindex(child_shape):
                    combined_items[sweep_idx + inner_idx] = node._items[inner_idx]

            result_dims = sweep_dims + child_dims
            result_coords = {**sweep_coords, **child_coords}
        elif not first.dims:
            for node in item_nodes[1:]:
                if node.dims:
                    raise ConfigurationError(
                        "Sweep cannot mix scalar and dimensioned instantiations."
                    )
            combined_items = np.array(item_nodes, dtype=object).reshape(sweep_shape)
            result_dims = sweep_dims
            result_coords = sweep_coords
        else:
            raise ConfigurationError("Unsupported sweep node layout.")

        sweep_signature = tuple(
            (path, tuple(_canonical(v) for v in values))
            for path, values, _ in axis_specs
        )

        def _sweep_root() -> None:
            return None

        _sweep_root.__name__ = f"instantiate_sweep_{name}"
        root = Node(
            fn=_sweep_root,
            inputs={"_sweep_signature": sweep_signature},
            cache=True,
            dims=result_dims,
            coords=result_coords,
        )
        root._items = combined_items
        root._exec_deps = list(root.deps_nodes)
        seen_hashes: set[int] = set()
        for dep in combined_items.flat:
            if isinstance(dep, Node) and dep._hash not in seen_hashes:
                seen_hashes.add(dep._hash)
                root._exec_deps.append(dep)
        return root

    def instantiate(
        self,
        name: str,
        *,
        runtime: Any | None = None,
        sweep: Mapping[str, Any] | None = None,
    ) -> "Node":
        """Instantiate a node from config section ``name``.

        Parameters
        ----------
        name:
            Name of the config section to instantiate.
        runtime:
            Runtime used to resolve nested ``${...}`` references to nodes.
            If omitted, use the global runtime.
        sweep:
            Optional parameter sweep mapping. Keys are config paths (supports dot
            notation such as ``"trop_ls.degree"``) and values are candidate
            collections. Passing sweep returns a dimensioned node covering the
            Cartesian product of sweep values.

        Notes
        -----
        ``instantiate()`` binds parameters from the current config at call time.
        The returned node will not dynamically re-read later ``node.cfg`` edits.
        After mutating config values, call ``instantiate()`` again to build a node
        with the updated configuration snapshot.
        """
        from .exceptions import ConfigurationError

        if runtime is None:
            from .runtime import get_runtime

            runtime = get_runtime()

        if sweep is not None:
            if not isinstance(sweep, Mapping):
                raise ConfigurationError("sweep must be a mapping of path -> values.")

        if name not in self._conf:
            raise ConfigurationError(f"Config section '{name}' not found.")

        raw_cfg = self._conf[name]
        if not OmegaConf.is_dict(raw_cfg):
            raise ConfigurationError(
                f"Config section '{name}' must be a mapping, got {type(raw_cfg).__name__}."
            )

        cfg = self._resolve_with_presets(cast(DictConfig, raw_cfg))
        target = cfg.get("_target_", name)
        if not isinstance(target, str) or not target.strip():
            raise ConfigurationError(
                f"Config section '{name}' has invalid '_target_' value: {target!r}."
            )

        try:
            fn = self._locate(target)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to locate target '{target}' for section '{name}': {e}"
            ) from e

        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        if not isinstance(cfg_dict, dict):
            raise ConfigurationError(
                f"Config section '{name}' must resolve to a mapping."
            )

        params = {
            k: self._resolve_value(v, runtime)
            for k, v in cfg_dict.items()
            if k not in {"_target_", "_use_", "_presets_"}
        }
        return (
            self._instantiate_with_sweep(name=name, fn=fn, params=params, sweep=sweep)
            if sweep
            else fn(**params)
        )

    def _build_node(self, name: str, runtime: Any) -> "Node":
        return self.instantiate(name, runtime=runtime)

    def defaults(self, fn_name: str, *, runtime: Any | None = None) -> dict[str, Any]:
        node_cfg = self._conf.get(fn_name)
        if node_cfg is None:
            return {}
        
        # Resolve presets if present
        node_cfg = self._resolve_with_presets(node_cfg)
        
        if runtime is None:
            return cast(dict[str, Any], OmegaConf.to_container(node_cfg, resolve=True))
        result: dict[str, Any] = {}
        for k, v in OmegaConf.to_container(node_cfg, resolve=False).items():
            if k == "_target_":
                continue
            result[k] = self._resolve_value(v, runtime)
        return result

    def copy_from(self, other: "Config") -> None:
        """Copy ``other`` into this config without changing object identity."""
        for key in list(self._conf.keys()):
            del self._conf[key]
        data = OmegaConf.to_container(other._conf, resolve=False) or {}
        for key, value in data.items():
            self._conf[key] = value