"""
postproc/plot_config.py
=======================
Figure style settings for postprocess.py, read from configs/plots/.

configs/plots/default.yaml holds every setting. An optional override file
(--plot-config) is merged on top of it; it may only contain keys that exist
in default.yaml, so typos fail early instead of being ignored.

Plot functions receive a PlotConfig and use:
    st = style.figure(mode, name)     settings of one figure (defaults + its own)
    cb = style.colorbar(var, values)  cmap / norm / ticks for one mapped variable
    style.savefig(fig, path_no_ext)   save with the output settings, then close

Everything used while drawing is recorded, and write_record() saves it next to
the figures as plot_config_used.yaml (one entry per figure, so partial
redraws keep the record of the other figures).
"""

import copy
import datetime
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib as mpl
import matplotlib.backend_bases
import matplotlib.pyplot as plt
import yaml

from postproc.utils import compute_smart_bounds


DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "plots" / "default.yaml"

MODES   = ("discrete", "continuous")
EXTENDS = ("neither", "min", "max", "both")


class PlotConfigError(ValueError):
    pass


class _NoAliasDumper(yaml.SafeDumper):
    """Write repeated values in full instead of YAML anchors (&id001)."""
    def ignore_aliases(self, data):
        return True


# ==========================================================
# Loading and merging
# ==========================================================

def _check_keys(block, allowed, where):
    unknown = [k for k in (block or {}) if k not in allowed]
    if unknown:
        raise PlotConfigError(
            f"unknown key(s) {', '.join(f'{where}.{k}' for k in unknown)}. "
            f"Valid keys: {', '.join(sorted(allowed))}"
        )


def _merge(base, override, where):
    """Merge override into a copy of base; every key must already exist in base."""
    out = copy.deepcopy(base)
    _check_keys(override, base, where)
    for k, v in (override or {}).items():
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, f"{where}.{k}")
        else:
            out[k] = v
    return out


def _merge_config(default, override):
    """
    Merge an override file into default.yaml.

    figures.<mode>.<name> accepts the figure's own keys plus any key of
    defaults; colorbars.<var> accepts any key of colorbar_defaults.
    """
    out = copy.deepcopy(default)
    _check_keys(override, default, "plot config")
    for section, val in (override or {}).items():
        if section == "figures":
            _check_keys(val, default["figures"], "figures")
            for mode, figs in (val or {}).items():
                _check_keys(figs, default["figures"][mode], f"figures.{mode}")
                for name, block in (figs or {}).items():
                    allowed = set(default["defaults"]) | set(default["figures"][mode][name])
                    _check_keys(block, allowed, f"figures.{mode}.{name}")
                    out["figures"][mode][name].update(block or {})
        elif section == "colorbars":
            _check_keys(val, default["colorbars"], "colorbars")
            for var, block in (val or {}).items():
                allowed = set(default["colorbar_defaults"]) | set(default["colorbars"][var])
                _check_keys(block, allowed, f"colorbars.{var}")
                out["colorbars"][var].update(block or {})
        elif isinstance(default[section], dict):
            out[section] = _merge(default[section], val, section)
        else:
            out[section] = val
    return out


def load_plot_config(path=None):
    """Load default.yaml, merge the optional override file and validate."""
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sources = [os.path.relpath(DEFAULT_PATH)]

    if path is not None and Path(path).resolve() != DEFAULT_PATH:
        if not os.path.exists(path):
            raise PlotConfigError(f"plot config not found: {path}")
        with open(path, encoding="utf-8") as f:
            cfg = _merge_config(cfg, yaml.safe_load(f))
        sources.append(str(path))

    _validate(cfg)
    return PlotConfig(cfg, sources)


# ==========================================================
# Validation
# ==========================================================

def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate_colorbar(var, spec):
    where = f"colorbars.{var}"
    if spec.get("cmap") not in mpl.colormaps:
        raise PlotConfigError(f"{where}.cmap: unknown matplotlib colormap '{spec.get('cmap')}'")
    if spec["mode"] not in MODES:
        raise PlotConfigError(f"{where}.mode must be one of {MODES}, got '{spec['mode']}'")
    if spec["extend"] not in EXTENDS:
        raise PlotConfigError(f"{where}.extend must be one of {EXTENDS}, got '{spec['extend']}'")
    if spec["symmetric"] not in ("auto", True, False):
        raise PlotConfigError(f"{where}.symmetric must be auto, true or false")
    if not isinstance(spec["n_bins"], int) or spec["n_bins"] < 1:
        raise PlotConfigError(f"{where}.n_bins must be a positive integer")

    b = spec["bounds"]
    if isinstance(b, list):
        if len(b) < 2 or not all(_is_number(x) for x in b) or np.any(np.diff(b) <= 0):
            raise PlotConfigError(f"{where}.bounds must be at least 2 strictly increasing numbers")
    elif isinstance(b, dict):
        if set(b) != {"vmin", "vmax", "step"} or not all(_is_number(x) for x in b.values()):
            raise PlotConfigError(f"{where}.bounds must be {{vmin: ., vmax: ., step: .}}")
        if b["vmax"] <= b["vmin"] or b["step"] <= 0:
            raise PlotConfigError(f"{where}.bounds needs vmin < vmax and step > 0")
    elif b != "auto":
        raise PlotConfigError(f"{where}.bounds must be auto, a list of edges or {{vmin, vmax, step}}")

    q = spec["auto_quantiles"]
    if q is not None and not (isinstance(q, list) and len(q) == 2
                              and all(_is_number(x) for x in q) and 0 <= q[0] < q[1] <= 100):
        raise PlotConfigError(f"{where}.auto_quantiles must be null or [low, high] with 0 <= low < high <= 100")
    for k in ("clip_min", "clip_max"):
        if spec[k] is not None and not _is_number(spec[k]):
            raise PlotConfigError(f"{where}.{k} must be null or a number")
    if spec["tick_format"] is not None:
        try:
            spec["tick_format"].format(1.0)
        except Exception:
            raise PlotConfigError(f"{where}.tick_format must be a format string such as '{{:.1f}}'")


def _validate(cfg):
    out = cfg["output"]
    supported = mpl.backend_bases.FigureCanvasBase.get_supported_filetypes()
    if out["format"] not in supported:
        raise PlotConfigError(f"output.format must be one of {', '.join(sorted(supported))}")
    if not _is_number(out["dpi"]) or out["dpi"] <= 0:
        raise PlotConfigError("output.dpi must be a positive number")

    for var in cfg["colorbars"]:
        _validate_colorbar(var, {**cfg["colorbar_defaults"], **cfg["colorbars"][var]})

    colors = cfg["colors"]
    flat = [(k, v) for k, v in colors.items() if isinstance(v, str)]
    flat += [(f"seasons.{k}", v) for k, v in colors["seasons"].items()]
    flat += [(f"regions.{k}", v) for k, v in colors["regions"].items()]
    flat += [(f"models[{i}]", v) for i, v in enumerate(colors["models"])]
    for k, v in flat:
        if not mpl.colors.is_color_like(v):
            raise PlotConfigError(f"colors.{k}: '{v}' is not a matplotlib color")
    if not colors["models"]:
        raise PlotConfigError("colors.models must contain at least one color")


# ==========================================================
# PlotConfig
# ==========================================================

def _resolve_bounds(spec, values):
    b = spec["bounds"]
    if isinstance(b, list):
        return [float(x) for x in b]
    if isinstance(b, dict):
        n = int(round((b["vmax"] - b["vmin"]) / b["step"]))
        return [round(b["vmin"] + i * b["step"], 10) for i in range(n + 1)]

    bounds = compute_smart_bounds(values, n_bins=spec["n_bins"],
                                  quantiles=spec["auto_quantiles"],
                                  symmetric=spec["symmetric"])
    lo, hi = spec["clip_min"], spec["clip_max"]
    if hi is not None:
        bounds = [x for x in bounds if x <= hi]
        if not bounds:
            bounds = [hi - 1.0, hi]
        elif bounds[-1] < hi:
            bounds.append(hi)
    if lo is not None:
        bounds = [x for x in bounds if x >= lo]
        if not bounds:
            bounds = [lo, lo + 1.0]
        elif bounds[0] > lo:
            bounds.insert(0, lo)
    if len(bounds) == 1:
        bounds.insert(0, bounds[0] - 1.0)
    return [float(x) for x in bounds]


class PlotConfig:
    def __init__(self, cfg, sources):
        self.cfg     = cfg
        self.sources = sources
        self.records = {}
        self._current = None

    @property
    def colors(self):
        return self.cfg["colors"]

    def model_colors(self, n):
        palette = self.cfg["colors"]["models"]
        return [palette[i % len(palette)] for i in range(n)]

    def figure(self, mode, name):
        """Settings of one figure: defaults overridden by the figure's own keys."""
        st = {**self.cfg["defaults"], **self.cfg["figures"][mode][name]}
        self._current = {"mode": mode, "settings": copy.deepcopy(st), "colorbars": {}, "files": []}
        self.records[name] = self._current
        st["dpi"] = self.cfg["output"]["dpi"]
        return SimpleNamespace(**st)

    def colorbar(self, var, values=None):
        """
        Colormap, norm and ticks for one variable.

        values are the data the colorbar covers (all models together in
        intercomparison maps); they are only used when bounds is auto.
        """
        spec   = {**self.cfg["colorbar_defaults"], **self.cfg["colorbars"][var]}
        bounds = _resolve_bounds(spec, values)
        extend = spec["extend"]

        if spec["mode"] == "discrete":
            n_ext = int(extend in ("min", "both")) + int(extend in ("max", "both"))
            n     = len(bounds) - 1 + n_ext
            cmap  = mpl.colors.ListedColormap(
                mpl.colormaps[spec["cmap"]].resampled(n)(np.linspace(0, 1, n))
            )
            norm  = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=n, extend=extend)
        else:
            cmap  = mpl.colormaps[spec["cmap"]]
            norm  = mpl.colors.Normalize(vmin=bounds[0], vmax=bounds[-1])

        if self._current is not None:
            self._current["colorbars"][var] = {
                "cmap": spec["cmap"], "mode": spec["mode"],
                "bounds": bounds, "extend": extend,
            }
        return SimpleNamespace(label=spec["label"], cmap=cmap, norm=norm, bounds=bounds,
                               ticks=bounds, extend=extend, tick_format=spec["tick_format"])

    def savefig(self, fig, path_no_ext):
        """Save fig as <path_no_ext>.<format> and close it. Returns the file name."""
        out   = self.cfg["output"]
        fname = f"{path_no_ext}.{out['format']}"
        fig.savefig(fname, dpi=out["dpi"], bbox_inches="tight",
                    transparent=out["transparent"])
        plt.close(fig)
        if self._current is not None:
            self._current["files"].append(fname)
        return fname

    def write_record(self, out_dir):
        """Merge the settings of the figures drawn in this run into plot_config_used.yaml."""
        path = os.path.join(out_dir, "plot_config_used.yaml")
        data = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        figures = data.get("figures") or {}

        now = datetime.datetime.now().isoformat(timespec="seconds")
        for name, rec in self.records.items():
            figures[name] = {
                "drawn":       now,
                "plot_config": self.sources,
                "output":      copy.deepcopy(self.cfg["output"]),
                "files":       [os.path.relpath(f, out_dir) for f in rec["files"]],
                "settings":    rec["settings"],
                "colorbars":   rec["colorbars"],
            }

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Written by postprocess.py: the exact style used for each figure.\n")
            yaml.dump({"figures": figures}, f, Dumper=_NoAliasDumper,
                      sort_keys=False, allow_unicode=True)
        self.records = {}
        self._current = None
        return path


_DEFAULT = None


def default_style():
    """PlotConfig from default.yaml, loaded once (for plot functions called without one)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load_plot_config()
    return _DEFAULT
