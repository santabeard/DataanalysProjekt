#!/usr/bin/env python3
"""Helpers that turn exported missingness parameters into synthetic masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VITAL_VARIABLES = ["sbp", "pulse", "rr", "spo2", "temp"]
VITAL_CONCEPTS = {
    "sbp": "systolic_bp", "pulse": "pulse_rate", "rr": "resp_rate",
    "spo2": "spo2", "temp": "body_temp",
}


def _read(stats_dir: str | Path, name: str) -> pd.DataFrame:
    path = Path(stats_dir) / f"{name}.csv"
    df = pd.read_csv(path, dtype={"pattern": str}) if path.exists() else pd.DataFrame()
    for col in ("domain", "group", "source", "coef", "variable"):
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    return df


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -30, 30)
    return 1 / (1 + np.exp(-x))


def add_base_design(df: pd.DataFrame, complaints: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["age"] = df["ed_age"]
    df["male"] = df["is_male"]
    code = df["ed_presented_complaint_std"].astype("string").str.extract(r"\[(\d+)\]", expand=False)
    for c in complaints:
        df[f"sct_{c}"] = (code == str(c)).astype(float)
    return df


class MissingnessModel:
    def __init__(self, stats_dir: str | Path, domain: str, variables: list[str], pattern_file: str,
                 source: str | None = None):
        self.domain, self.variables, self.source = domain, list(variables), source
        observed = _read(stats_dir, "missingness_by_hospital")
        self.patterns = _read(stats_dir, pattern_file)
        selection = _read(stats_dir, "missingness_selection")
        self.observed = observed[(observed["domain"] == domain)] if not observed.empty else observed
        if not self.patterns.empty and "domain" in self.patterns:
            self.patterns = self.patterns[self.patterns["domain"] == domain]
        if source is not None:
            for frame_name in ("observed", "patterns"):
                frame = getattr(self, frame_name)
                if not frame.empty and "source" in frame:
                    values = frame["source"].fillna("").astype(str)
                    setattr(self, frame_name, frame[values == source])
        if not selection.empty:
            selection = selection[selection["domain"] == domain]
            if source is not None and "source" in selection:
                selection = selection[selection["source"].fillna("").astype(str) == source]
            models = set(selection.get("model", []))
            self.selection = selection[selection["model"] == ("MNAR" if "MNAR" in models else "MAR")]
        else:
            self.selection = selection
        self.available = not self.selection.empty and not self.patterns.empty
        self.coef: dict[tuple, float] = {}
        self.meta: dict[tuple, dict] = {}
        self.base_scale: dict[tuple, tuple[float, float]] = {}
        self.base_labels: set[str] = set()
        self.sel_hosp_labels: set[str] = set()
        if not self.selection.empty:
            for row in self.selection.itertuples(index=False):
                key = (str(row.group), str(row.source), str(row.variable))
                self.meta[key] = {"x_loc": float(row.x_loc), "x_scale": float(row.x_scale),
                                  "log": int(row.log), "rate": float(row.observed_rate)}
                self.coef[(*key, str(row.coef))] = float(row.mean)
                if str(row.coef).startswith("sel_hosp_"):
                    self.sel_hosp_labels.add(str(row.coef))
                if str(row.coef) in ("age", "male") or str(row.coef).startswith("sct_"):
                    self.base_labels.add(str(row.coef))
                    if pd.notna(row.z_loc) and pd.notna(row.z_scale):
                        self.base_scale[(key[0], key[1], str(row.coef))] = (float(row.z_loc), float(row.z_scale))
        self.rates: dict[tuple, float] = {}
        if not self.observed.empty:
            for _, row in self.observed.iterrows():
                key = (str(row["group"]), str(row["source"]), str(row["variable"]))
                value = row["observed_rate"] if "observed_rate" in row else row[row["variable"]]
                self.rates[key] = float(value)
        self.patterns_by_key: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        if not self.patterns.empty:
            for key, g in self.patterns.groupby(["group", "source"], dropna=False):
                pats = g["pattern"].astype(str).tolist()
                counts = g["n"].to_numpy(dtype=float)
                keep = counts > 0
                if not keep.any():
                    continue
                matrix = np.array([[int(ch) for ch in p] for p, ok in zip(pats, keep) if ok], dtype=float)
                if matrix.shape[1] == len(self.variables):
                    self.patterns_by_key[(str(key[0]), str(key[1]))] = (matrix, counts[keep])

    def _keys(self, group: str, source: str, variable: str) -> list[tuple[str, str, str]]:
        group, source = str(group), str(source)
        keys = [(group, source, variable)]
        for fallback in ((group, ""), ("", source), ("", "")):
            if fallback not in keys:
                keys.append(fallback)
        return keys

    def _coef(self, group: str, source: str, variable: str, label: str, default: float = 0.0) -> float:
        for key in self._keys(group, source, variable):
            if (*key, label) in self.coef:
                return self.coef[(*key, label)]
        return default

    def _meta(self, group: str, source: str, variable: str) -> dict | None:
        for key in self._keys(group, source, variable):
            if key in self.meta:
                return self.meta[key]
        return None

    def _scale(self, group: str, source: str, variable: str, label: str) -> tuple[float, float]:
        for key in self._keys(group, source, variable):
            scale_key = (key[0], key[1], label)
            if scale_key in self.base_scale:
                return self.base_scale[scale_key]
        return 0.0, 1.0

    def _eta(self, frame: pd.DataFrame, latent: pd.DataFrame, group: str, source: str, idx: np.ndarray,
             gx_scale: float) -> np.ndarray:
        n, p = len(idx), len(self.variables)
        eta = np.zeros((n, p))
        hospital = str(group).replace("-", "_")
        died = frame["died30"].to_numpy(dtype=float)[idx] if "died30" in frame else np.zeros(n)
        for j, name in enumerate(self.variables):
            eta[:, j] += self._coef(group, source, name, "intercept")
            eta[:, j] += self._coef(group, source, name, "outcome") * died
            for label in self.sel_hosp_labels:
                if label.replace("sel_hosp_", "", 1) == hospital:
                    eta[:, j] += self._coef(group, source, name, label)
            for label in sorted(self.base_labels):
                if label not in frame:
                    continue
                loc, scale = self._scale(group, source, name, label)
                z = (frame[label].to_numpy(dtype=float)[idx] - loc) / max(scale, 1e-9)
                eta[:, j] += self._coef(group, source, name, label) * z
            meta = self._meta(group, source, name)
            if meta:
                x = latent[name].to_numpy(dtype=float)[idx]
                if meta["log"]:
                    x = np.log(np.clip(x, 1e-6, None))
                zx = (x - meta["x_loc"]) / max(meta["x_scale"], 1e-9)
                eta[:, j] += self._coef(group, source, name, "own_value") * gx_scale * zx
        return eta

    def _target_rates(self, group: str, source: str) -> np.ndarray:
        out = []
        for name in self.variables:
            key = (group, source, name)
            meta = self._meta(group, source, name)
            if key in self.rates:
                out.append(self.rates[key])
            elif meta:
                out.append(meta["rate"])
            else:
                out.append(0.0)
        return np.clip(np.asarray(out, dtype=float), 1e-4, 1 - 1e-4)

    def sample(self, frame: pd.DataFrame, latent: pd.DataFrame, rng: np.random.Generator,
               gx_scale: float = 1.0, group_col: str = "ed_hospital",
               source_col: str | None = None) -> pd.DataFrame:
        out = pd.DataFrame(False, index=frame.index, columns=self.variables)
        source_values = frame[source_col].fillna("").astype(str).to_numpy() if source_col in frame else \
            np.array([""] * len(frame))
        groups = list(zip(frame[group_col].astype(str).to_numpy(), source_values))
        for group, source in sorted(set(groups)):
            idx = np.flatnonzero(np.asarray([k == (group, source) for k in groups], dtype=bool))
            eta = self._eta(frame, latent, group, source, idx, gx_scale)
            target = self._target_rates(group, source)
            info = self.patterns_by_key.get((group, source))
            if info is None:
                adjusted = _logit(target)[None, :] - _logit(_expit(eta).mean(axis=0))[None, :]
                probs = _expit(eta + adjusted)
                out.iloc[idx] = rng.random(eta.shape) < probs
                continue
            patterns, counts = info
            delta = np.zeros(len(self.variables))
            log_counts = np.log(np.maximum(counts, 1e-9))
            for _ in range(25):
                lw = (eta + delta) @ patterns.T + log_counts[None, :]
                lw -= np.nanmax(lw, axis=1, keepdims=True)
                probs = np.exp(lw)
                probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
                marginal = probs @ patterns
                change = (_logit(target)[None, :] - _logit(marginal)).mean(axis=0)
                delta = np.clip(delta + 0.5 * np.clip(change, -1, 1), -15, 15)
            lw = (eta + delta) @ patterns.T + log_counts[None, :]
            lw -= np.nanmax(lw, axis=1, keepdims=True)
            probs = np.exp(lw)
            probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
            choices = [rng.choice(len(patterns), p=probs[i]) for i in range(len(idx))]
            out.iloc[idx] = patterns[np.asarray(choices)].astype(bool)
        return out
