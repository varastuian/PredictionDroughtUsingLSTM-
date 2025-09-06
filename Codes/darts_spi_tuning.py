#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darts LSTM hyperparameter search per station for SPI targets.

Usage (example):
    python darts_spi_tuning.py --csv your_file.csv --datefmt "%d/%m/%Y" \
        --targets SPI_1 SPI_3 SPI_6 --n-trials 40 --horizon 1 --min-train-fraction 0.6 \
        --seed 7 --outdir results_spi_tuning

Notes:
- Expects columns: station_id, ds, [targets...], plus optional regressors like precip, tm_m.
- Works with monthly data. If your data are daily but SPI is monthly, resample before using.
- Requires: darts>=0.30, torch, pandas, numpy
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import rmse, mae, mape
from darts.models import RNNModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
# from darts.utils import logging as darts_logging
from darts.utils.statistics import check_seasonality

import torch

# darts_logging.get_logger("darts").setLevel("ERROR")

@dataclass
class TrialResult:
    station_id: str
    target: str
    params: Dict
    rmse: float
    mae: float
    mape: float

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_series_per_station(df: pd.DataFrame,
                             targets: List[str],
                             regressors: Optional[List[str]] = None,
                             date_col: str = "ds",
                             station_col: str = "station_id",
                             datefmt: Optional[str] = None) -> Dict[str, Dict[str, TimeSeries]]:
    """Returns nested dict: station_id -> {"y:<target>": series, "X": past_covariates}"""
    # Parse dates
    if datefmt:
        df[date_col] = pd.to_datetime(df[date_col], format=datefmt, errors="coerce")
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([station_col, date_col]).dropna(subset=[date_col])

    stations = {}
    for sid, g in df.groupby(station_col):
        g = g.set_index(date_col)
        station_dict = {}
        # Targets
        for t in targets:
            if t not in g.columns:
                continue
            ser = TimeSeries.from_series(g[t].astype(float), fill_missing_dates=True, freq=None)
            station_dict[f"y:{t}"] = ser
        # Past covariates (optionally)
        X = None
        if regressors:
            covs = []
            for c in regressors:
                if c in g.columns:
                    covs.append(TimeSeries.from_series(g[c].astype(float), fill_missing_dates=True, freq=None))
            if covs:
                X = covs[0]
                for c in covs[1:]:
                    X = X.stack(c)
        # Calendar features (month-of-year) can help
        if len(g) > 2:
            cal = datetime_attribute_timeseries(
                pd.date_range(g.index.min(), g.index.max(), freq=pd.infer_freq(g.index) or "MS"),
                attribute="month",
                one_hot=True
            )
            X = cal if X is None else X.stack(cal)

        if X is not None:
            station_dict["X"] = X

        if station_dict:
            stations[str(sid)] = station_dict

    return stations

def train_val_split(series: TimeSeries, min_train_fraction: float = 0.7) -> Tuple[TimeSeries, TimeSeries]:
    split_idx = int(len(series) * min_train_fraction)
    return series[:split_idx], series[split_idx:]

def sample_params() -> Dict:
    # Reasonable ranges for monthly SPI
    return {
        "hidden_dim": random.choice([16, 24, 32, 48, 64, 96, 128]),
        "n_rnn_layers": random.choice([1, 2, 3]),
        "dropout": random.choice([0.0, 0.05, 0.1, 0.2, 0.3]),
        "batch_size": random.choice([16, 32, 64, 128]),
        "n_epochs": random.choice([100, 150, 200]),
        "optimizer_kwargs": {"lr": random.choice([1e-4, 3e-4, 1e-3, 3e-3])},
        # Try enabling/disable training length truncation
        "training_length": random.choice([12, 18, 24, 36, 48]),
        "model_name": None,  # set later
        "random_state": random.randint(1, 10_000)
    }

def evaluate_one(series: TimeSeries,
                 past_covs: Optional[TimeSeries],
                 params: Dict,
                 horizon: int = 1,
                 val_stride: int = 1,
                 start_fraction: float = 0.6,
                 device: str = "cpu") -> Tuple[float, float, float]:
    # Scale target (and covariates) for stability
    y_scaler = Scaler()
    s = y_scaler.fit_transform(series)
    X = None
    if past_covs is not None:
        x_scaler = Scaler()
        X = x_scaler.fit_transform(past_covs)
    filler = MissingValuesFiller()
    s = filler.transform(s)
    if X is not None:
        X = filler.transform(X)

    model = RNNModel(
        model="LSTM",
        input_chunk_length=params["training_length"],
        output_chunk_length=horizon,
        hidden_dim=params["hidden_dim"],
        n_rnn_layers=params["n_rnn_layers"],
        dropout=params["dropout"],
        batch_size=params["batch_size"],
        n_epochs=params["n_epochs"],
        optimizer_kwargs=params["optimizer_kwargs"],
        model_name=params["model_name"],
        log_tensorboard=False,
        random_state=params["random_state"],
        pl_trainer_kwargs={
            "accelerator": "gpu" if device == "cuda" else "cpu",
            "devices": 1 if device == "cuda" else None,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        save_checkpoints=False,
        force_reset=True,
    )

    # Historical forecasts backtest
    preds = model.historical_forecasts(
        s,
        past_covariates=X,
        start=int(len(s) * start_fraction),
        forecast_horizon=horizon,
        stride=val_stride,
        retrain=True,  # retrain at each step on expanding window (robust but slower)
        verbose=False,
    )

    r = rmse(s, preds)
    a = mae(s, preds)
    p = mape(s, preds)
    return float(r), float(a), float(p)

def run_search_per_target(station_id: str,
                          y: TimeSeries,
                          X: Optional[TimeSeries],
                          n_trials: int,
                          horizon: int,
                          min_train_fraction: float,
                          device: str) -> TrialResult:
    best: Optional[TrialResult] = None
    for i in range(1, n_trials + 1):
        params = sample_params()
        params["model_name"] = f"lstm_{station_id}_{i}"
        try:
            r, a, p = evaluate_one(y, X, params, horizon=horizon, start_fraction=min_train_fraction, device=device)
        except Exception as e:
            # skip failed trials
            print(f"[{station_id}] Trial {i} failed: {e}")
            continue
        tr = TrialResult(station_id=station_id, target=y.components[0], params=params, rmse=r, mae=a, mape=p)
        if (best is None) or (r < best.rmse):
            best = tr
        print(f"[{station_id}] Trial {i}/{n_trials} -> RMSE={r:.4f} MAE={a:.4f} MAPE={p:.2f}% | best RMSE={(best.rmse if best else np.inf):.4f}")
    if best is None:
        raise RuntimeError(f"No successful trials for station {station_id}")
    return best

def fit_final_model(series: TimeSeries,
                    past_covs: Optional[TimeSeries],
                    best_params: Dict,
                    horizon: int,
                    device: str) -> RNNModel:
    y_scaler = Scaler()
    s = y_scaler.fit_transform(series)
    X = None
    if past_covs is not None:
        x_scaler = Scaler()
        X = x_scaler.fit_transform(past_covs)
    filler = MissingValuesFiller()
    s = filler.transform(s)
    if X is not None:
        X = filler.transform(X)

    model = RNNModel(
        model="LSTM",
        input_chunk_length=best_params["training_length"],
        output_chunk_length=horizon,
        hidden_dim=best_params["hidden_dim"],
        n_rnn_layers=best_params["n_rnn_layers"],
        dropout=best_params["dropout"],
        batch_size=best_params["batch_size"],
        n_epochs=best_params["n_epochs"],
        optimizer_kwargs=best_params["optimizer_kwargs"],
        model_name="final_" + best_params.get("model_name", "lstm"),
        log_tensorboard=False,
        random_state=best_params.get("random_state", 42),
        pl_trainer_kwargs={
            "accelerator": "gpu" if device == "cuda" else "cpu",
            "devices": 1 if device == "cuda" else None,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
        save_checkpoints=False,
        force_reset=True,
    )
    model.fit(series=s, past_covariates=X, verbose=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=r"Data\python_spi\40706.csv", help="Path to CSV with columns [station_id, ds, targets...]")
    parser.add_argument("--datefmt", type=str, default=None, help="Date format for 'ds' (e.g., '%%d/%%m/%%Y')")
    parser.add_argument("--targets", type=str, nargs="+", default=["SPI_12"], help="Target columns to model")
    parser.add_argument("--regressors", type=str, nargs="*", default=["precip", "tm_m"], help="Optional past covariates")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--min-train-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="spi_tuning_results")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    set_global_seed(args.seed)
    device = pick_device()
    print(f"Using device: {device}")

    df = pd.read_csv(args.csv)
    stations = build_series_per_station(
        df,
        targets=args.targets,
        regressors=args.regressors,
        date_col="ds",
        station_col="station_id",
        datefmt=args.datefmt
    )

    all_results: List[TrialResult] = []
    best_by_station_target: Dict[Tuple[str, str], TrialResult] = {}

    for sid, bundle in stations.items():
        X = bundle.get("X", None)
        for key, y in bundle.items():
            if not key.startswith("y:"):
                continue
            target_name = key.split("y:")[1]
            print(f"\n=== Station {sid} | Target {target_name} ===")
            best = run_search_per_target(
                station_id=sid,
                y=y,
                X=X,
                n_trials=args.n_trials,
                horizon=args.horizon,
                min_train_fraction=args.min_train_fraction,
                device=device,
            )
            all_results.append(best)
            best_by_station_target[(sid, target_name)] = best

    # Save results
    rows = []
    for tr in all_results:
        row = {
            "station_id": tr.station_id,
            "target": tr.target,
            "rmse": tr.rmse,
            "mae": tr.mae,
            "mape": tr.mape,
            **{f"param_{k}": v for k, v in tr.params.items()}
        }
        rows.append(row)
    res_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "tuning_results.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to: {csv_path}")

    best_dict = {
        f"{sid}:{t}": {
            "rmse": br.rmse,
            "mae": br.mae,
            "mape": br.mape,
            "params": br.params
        }
        for (sid, t), br in best_by_station_target.items()
    }
    json_path = os.path.join(args.outdir, "best_params.json")
    with open(json_path, "w") as f:
        json.dump(best_dict, f, indent=2)
    print(f"Saved best params to: {json_path}")

    # (Optional) retrain final models on full data and save torch checkpoints
    for (sid, t), br in best_by_station_target.items():
        y = stations[sid][f"y:{t}"]
        X = stations[sid].get("X", None)
        print(f"Training final model for station {sid}, target {t} ...")
        model = fit_final_model(y, X, br.params, horizon=args.horizon, device=device)
        ckpt_dir = os.path.join(args.outdir, "final_models")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save(os.path.join(ckpt_dir, f"final_lstm_{sid}_{t}.pth"))
    print("Done.")

if __name__ == "__main__":
    main()
