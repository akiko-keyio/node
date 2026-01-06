"""Tests for configuration preset functionality."""

import pytest
from node import Config


class TestConfigPresets:
    """Test cases for _presets_ and _use_ configuration."""

    def test_preset_resolution_basic(self):
        """Test basic preset resolution."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "freq": "12h",
                "_use_": "full",
                "_presets_": {
                    "month1": {"start": "2023-01-01", "end": "2023-01-31"},
                    "full": {"start": "2023-01-01", "end": "2023-12-31"},
                }
            }
        })
        
        defaults = cfg.defaults("times")
        assert defaults["_target_"] == "my.module.times"
        assert defaults["freq"] == "12h"
        assert defaults["start"] == "2023-01-01"
        assert defaults["end"] == "2023-12-31"

    def test_preset_switching(self):
        """Test switching between presets."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "freq": "12h",
                "_use_": "full",
                "_presets_": {
                    "month1": {"start": "2023-01-01", "end": "2023-01-31"},
                    "full": {"start": "2023-01-01", "end": "2023-12-31"},
                }
            }
        })
        
        # Initial: full preset
        defaults = cfg.defaults("times")
        assert defaults["end"] == "2023-12-31"
        
        # Switch to month1
        cfg._conf.times._use_ = "month1"
        defaults = cfg.defaults("times")
        assert defaults["end"] == "2023-01-31"
        # Base config should still be present
        assert defaults["freq"] == "12h"

    def test_no_preset_backward_compat(self):
        """Test backward compatibility for configs without presets."""
        cfg = Config({
            "simple": {
                "_target_": "my.module.simple",
                "x": 1,
                "y": 2,
            }
        })
        
        defaults = cfg.defaults("simple")
        assert defaults["_target_"] == "my.module.simple"
        assert defaults["x"] == 1
        assert defaults["y"] == 2

    def test_preset_overrides_target(self):
        """Test that preset can override _target_."""
        cfg = Config({
            "process": {
                "_target_": "module.v1",
                "threads": 4,
                "_use_": "fast",
                "_presets_": {
                    "normal": {},
                    "fast": {"_target_": "module.v2", "threads": 8},
                }
            }
        })
        
        defaults = cfg.defaults("process")
        assert defaults["_target_"] == "module.v2"
        assert defaults["threads"] == 8

    def test_preset_no_use_returns_base(self):
        """Test that without _use_, only base config is returned."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "freq": "12h",
                "_presets_": {
                    "month1": {"start": "2023-01-01", "end": "2023-01-31"},
                }
            }
        })
        
        defaults = cfg.defaults("times")
        assert defaults["_target_"] == "my.module.times"
        assert defaults["freq"] == "12h"
        assert "start" not in defaults
        assert "end" not in defaults

    def test_preset_not_found_returns_base(self):
        """Test that invalid preset name returns only base config."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "freq": "12h",
                "_use_": "nonexistent",
                "_presets_": {
                    "month1": {"start": "2023-01-01", "end": "2023-01-31"},
                }
            }
        })
        
        defaults = cfg.defaults("times")
        assert defaults["_target_"] == "my.module.times"
        assert defaults["freq"] == "12h"
        assert "start" not in defaults

    def test_preset_with_internal_values(self):
        """Test that preset values are correctly merged."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "base_value": 100,
                "_use_": "full",
                "_presets_": {
                    "full": {"preset_value": 200},
                }
            }
        })
        
        defaults = cfg.defaults("times")
        assert defaults["base_value"] == 100
        assert defaults["preset_value"] == 200

    def test_empty_base_only_presets(self):
        """Test config with no base, only presets."""
        cfg = Config({
            "processor": {
                "_use_": "mode1",
                "_presets_": {
                    "mode1": {"_target_": "a.b", "x": 1},
                    "mode2": {"_target_": "a.c", "x": 2},
                }
            }
        })
        
        defaults = cfg.defaults("processor")
        assert defaults["_target_"] == "a.b"
        assert defaults["x"] == 1


class TestConfigPresetsEdgeCases:
    """Edge case tests for preset configuration."""

    def test_empty_preset(self):
        """Test preset with empty dict."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "freq": "12h",
                "_use_": "default",
                "_presets_": {
                    "default": {},
                }
            }
        })
        
        defaults = cfg.defaults("times")
        assert defaults["_target_"] == "my.module.times"
        assert defaults["freq"] == "12h"

    def test_presets_not_filtered_from_result(self):
        """Test that _use_ and _presets_ are not in final result."""
        cfg = Config({
            "times": {
                "_target_": "my.module.times",
                "_use_": "full",
                "_presets_": {
                    "full": {"x": 1},
                }
            }
        })
        
        defaults = cfg.defaults("times")
        assert "_use_" not in defaults
        assert "_presets_" not in defaults
