"""Configuration management for default arguments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING, cast

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
        *,
        cache_nodes: bool = False,
    ) -> None:
        """Create a configuration mapping.

        Parameters
        ----------
        mapping:
            Initial configuration data.
        cache_nodes:
            When ``True`` reuse nodes built from this config to avoid repeated
            instantiation. Defaults to ``False``.
        """
        if isinstance(mapping, (str, Path)):
            try:
                object.__setattr__(self, "_conf", OmegaConf.load(str(mapping)))
            except Exception as e:
                from .exceptions import ConfigurationError
                raise ConfigurationError(f"Failed to load config from {mapping}: {e}") from e
        else:
            object.__setattr__(self, "_conf", OmegaConf.create(mapping or {}))
        object.__setattr__(self, "_cache_nodes", cache_nodes)
        object.__setattr__(self, "_nodes", {})

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

    def _build_node(self, name: str, runtime: Any) -> "Node":
        if self._cache_nodes and name in self._nodes:
            return self._nodes[name]
        cfg = self._conf[name]
        target = cfg.get("_target_", name)
        fn = self._locate(str(target))
        params = {
            k: self._resolve_value(v, runtime)
            for k, v in OmegaConf.to_container(cfg, resolve=False).items()
            if k != "_target_"
        }
        node = fn(**params)
        if self._cache_nodes:
            self._nodes[name] = node
        return node

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
        self._cache_nodes = other._cache_nodes
        self._nodes.clear()
        for key in list(self._conf.keys()):
            del self._conf[key]
        data = OmegaConf.to_container(other._conf, resolve=False) or {}
        for key, value in data.items():
            self._conf[key] = value