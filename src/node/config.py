"""Configuration management for default arguments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from .core import Node

__all__ = ["Config"]

_TOPLEVEL_SECTION = "__toplevel__"


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

    def _locate(self, path: str) -> Callable[..., Any]:
        mod_name, attr = path.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[attr])
        return getattr(mod, attr)

    def _resolve_value(
        self,
        val: Any,
        runtime: Any,
        *,
        memo: dict[tuple[int, str], "Node"] | None = None,
        overrides: Mapping[tuple[str, str], Any] | None = None,
    ) -> Any:
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            key = val[2:-1]
            if overrides and (_TOPLEVEL_SECTION, key) in overrides:
                return overrides[(_TOPLEVEL_SECTION, key)]
            if key in self._conf:
                cfg_val = self._conf[key]
                if OmegaConf.is_dict(cfg_val) and "_target_" in cfg_val:
                    return self._build_node(
                        key,
                        runtime,
                        memo=memo,
                        overrides=overrides,
                    )
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

    def _build_sweep_axis_node(self, dim_name: str, values: list[Any]) -> "Node":
        """Build one axis node used by instantiate(sweep=...)."""
        from .core import Node

        def _axis_stub() -> None:
            return None

        _axis_stub.__name__ = f"instantiate_sweep_dim_{dim_name}"
        _axis_stub._node_ignore = frozenset()  # type: ignore[attr-defined]
        # Keep axis values in inputs so node identity/cache key changes when
        # sweep values change across instantiate() calls.
        axis_signature = tuple(values)
        axis_node = Node(
            fn=_axis_stub,
            inputs={"_axis_values": axis_signature},
            cache=True,
            dims=(dim_name,),
            coords={dim_name: values},
        )
        axis_node._items = np.array(values, dtype=object)
        return axis_node

    def _build_sweep_overrides(
        self,
        sweep: Mapping[str, Any],
    ) -> dict[tuple[str, str], "Node"]:
        """Build section-param overrides from global sweep paths.

        Sweep keys may be either ``"section.param"`` (targeting a specific
        parameter inside a section) or a single top-level scalar name such as
        ``"ref_height"``.  Top-level scalar sweeps are stored with the sentinel
        section :data:`_TOPLEVEL_SECTION` and are resolved lazily in
        :meth:`_resolve_value` whenever ``${key}`` interpolation is encountered.
        """
        from .exceptions import ConfigurationError

        sentinel = object()
        dim_to_path: dict[str, str] = {}
        overrides: dict[tuple[str, str], "Node"] = {}
        for raw_path, raw_values in sweep.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ConfigurationError(
                    f"Invalid sweep key: {raw_path!r}. Expected non-empty string path."
                )
            path = raw_path.strip()
            parts = [p for p in path.split(".") if p]

            try:
                axis_values = self._normalize_sweep_axis_values(raw_values)
            except ValueError as e:
                raise ConfigurationError(f"Invalid sweep values for '{path}': {e}") from e

            if len(parts) == 1:
                key = parts[0]
                if key not in self._conf:
                    raise ConfigurationError(
                        f"Invalid sweep key '{path}': key not found in config."
                    )
                if OmegaConf.is_dict(self._conf[key]):
                    raise ConfigurationError(
                        f"Invalid sweep key '{path}': top-level key '{key}' is a "
                        "mapping, not a scalar. Use 'section.param' to sweep "
                        "parameters within a section."
                    )
                dim_name = f"sweep_{key}"
                previous = dim_to_path.get(dim_name)
                if previous is not None and previous != path:
                    raise ConfigurationError(
                        f"Sweep dimension name collision: '{previous}' and '{path}' "
                        f"both map to '{dim_name}'."
                    )
                dim_to_path[dim_name] = path
                overrides[(_TOPLEVEL_SECTION, key)] = self._build_sweep_axis_node(
                    dim_name, axis_values
                )
                continue

            section = parts[0]
            param_path = ".".join(parts[1:])
            if section not in self._conf or not OmegaConf.is_dict(self._conf[section]):
                raise ConfigurationError(
                    f"Invalid sweep key '{path}': section '{section}' is not a config mapping."
                )

            current = OmegaConf.select(self._conf[section], param_path, default=sentinel)
            if current is sentinel:
                raise ConfigurationError(f"Invalid sweep key '{path}': path does not exist.")

            leaf = parts[-1]
            dim_name = f"sweep_{leaf}"
            previous = dim_to_path.get(dim_name)
            if previous is not None and previous != path:
                raise ConfigurationError(
                    f"Sweep dimension name collision: '{previous}' and '{path}' both map to '{dim_name}'."
                )
            dim_to_path[dim_name] = path
            overrides[(section, param_path)] = self._build_sweep_axis_node(dim_name, axis_values)
        return overrides

    def _wrap_dict_as_node(
        self,
        *,
        section: str,
        path: str,
        mapping_inputs: dict[str, Any],
    ) -> "Node":
        """Wrap a mapping as a Node so nested sweep dimensions can broadcast."""
        from .core import Node

        def _mapping_builder(**kwargs: Any) -> dict[str, Any]:
            return dict(kwargs)

        params = [
            inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
            for name in mapping_inputs
        ]
        _mapping_builder._node_sig = inspect.Signature(parameters=params)  # type: ignore[attr-defined]
        _mapping_builder._node_ignore = frozenset()  # type: ignore[attr-defined]
        safe_path = path.replace(".", "_")
        _mapping_builder.__name__ = f"instantiate_sweep_map_{section}_{safe_path}"
        return Node(fn=_mapping_builder, inputs=mapping_inputs, cache=True)

    def _resolve_mapping_with_overrides(
        self,
        *,
        section: str,
        base_path: str,
        raw_mapping: Mapping[str, Any],
        runtime: Any,
        memo: dict[tuple[int, str], "Node"],
        overrides: Mapping[tuple[str, str], Any],
    ) -> Any:
        """Resolve mapping values while applying sweep overrides recursively."""
        from .core import Node

        resolved: dict[str, Any] = {}
        has_dimension = False
        for key, raw_value in raw_mapping.items():
            full_path = f"{base_path}.{key}" if base_path else key
            override = overrides.get((section, full_path))
            if override is not None:
                resolved_value = override
            elif isinstance(raw_value, Mapping):
                resolved_value = self._resolve_mapping_with_overrides(
                    section=section,
                    base_path=full_path,
                    raw_mapping=raw_value,
                    runtime=runtime,
                    memo=memo,
                    overrides=overrides,
                )
            else:
                resolved_value = self._resolve_value(
                    raw_value,
                    runtime,
                    memo=memo,
                    overrides=overrides,
                )
            resolved[key] = resolved_value
            if isinstance(resolved_value, Node) and resolved_value.dims:
                has_dimension = True

        if has_dimension:
            return self._wrap_dict_as_node(
                section=section,
                path=base_path,
                mapping_inputs=resolved,
            )
        return resolved

    def _instantiate_impl(
        self,
        name: str,
        *,
        runtime: Any,
        memo: dict[tuple[int, str], "Node"],
        overrides: Mapping[tuple[str, str], Any] | None = None,
    ) -> "Node":
        from .exceptions import ConfigurationError

        if name not in self._conf:
            raise ConfigurationError(f"Config section '{name}' not found.")

        raw_cfg = self._conf[name]
        if not OmegaConf.is_dict(raw_cfg):
            raise ConfigurationError(
                f"Config section '{name}' must be a mapping, got {type(raw_cfg).__name__}."
            )

        target = raw_cfg.get("_target_", name)
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

        cfg_dict = OmegaConf.to_container(raw_cfg, resolve=False)
        if not isinstance(cfg_dict, dict):
            raise ConfigurationError(
                f"Config section '{name}' must resolve to a mapping."
            )

        params: dict[str, Any] = {}
        for k, v in cfg_dict.items():
            if k == "_target_":
                continue
            if overrides and (name, k) in overrides:
                params[k] = overrides[(name, k)]
            elif isinstance(v, Mapping):
                params[k] = self._resolve_mapping_with_overrides(
                    section=name,
                    base_path=k,
                    raw_mapping=v,
                    runtime=runtime,
                    memo=memo,
                    overrides=overrides or {},
                )
            else:
                params[k] = self._resolve_value(
                    v,
                    runtime,
                    memo=memo,
                    overrides=overrides,
                )
        return fn(**params)

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
            Optional parameter sweep mapping. Keys are either global config
            paths (``"section.param"``, e.g. ``"trop_ls.degree"``) or top-level
            scalar names (e.g. ``"ref_height"``).  Values are candidate
            collections.  Passing sweep returns a dimensioned node covering the
            Cartesian product of sweep values.  Top-level scalar sweeps are
            propagated through ``${...}`` interpolation, so all nodes referencing
            the swept key share the same sweep dimension.

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
            overrides = self._build_sweep_overrides(sweep) if sweep else None
        else:
            overrides = None

        return self._instantiate_impl(
            name,
            runtime=runtime,
            memo={},
            overrides=overrides,
        )

    def _build_node(
        self,
        name: str,
        runtime: Any,
        *,
        memo: dict[tuple[int, str], "Node"] | None = None,
        overrides: Mapping[tuple[str, str], Any] | None = None,
    ) -> "Node":
        if memo is None:
            memo = {}
        key = (id(runtime), name)
        if key in memo:
            return memo[key]
        node = self._instantiate_impl(
            name,
            runtime=runtime,
            memo=memo,
            overrides=overrides,
        )
        memo[key] = node
        return node

    def defaults(
        self,
        fn_name: str,
        *,
        runtime: Any | None = None,
        selected_names: set[str] | None = None,
    ) -> dict[str, Any]:
        node_cfg = self._conf.get(fn_name)
        if node_cfg is None:
            return {}
        if runtime is None:
            return cast(dict[str, Any], OmegaConf.to_container(node_cfg, resolve=True))
        memo: dict[tuple[int, str], "Node"] = {}
        result: dict[str, Any] = {}
        raw_items = OmegaConf.to_container(node_cfg, resolve=False)
        if not isinstance(raw_items, dict):
            return result
        for k, v in raw_items.items():
            if k == "_target_":
                continue
            if selected_names is not None and k not in selected_names:
                continue
            result[k] = self._resolve_value(v, runtime, memo=memo, overrides=None)
        return result

    def copy_from(self, other: "Config") -> None:
        """Copy ``other`` into this config without changing object identity."""
        for key in list(self._conf.keys()):
            del self._conf[key]
        data = OmegaConf.to_container(other._conf, resolve=False) or {}
        for key, value in data.items():
            self._conf[key] = value