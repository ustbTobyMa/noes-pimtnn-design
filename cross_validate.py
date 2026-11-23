"""Run five-fold cross-validation for magnetic and mechanical property prediction."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from dataset import DATA, FEATURES, TABLE_S4_DIGEST, TARGETS, read_records, record_digest
from model import MultiTaskNet, directional_penalty, observed_mse


DEFAULT_SEED = 42
TARGET_UNITS = ("T", "W/kg", "MPa")


def scaled_inputs(train, test):
    """Omit columns entirely missing in this training fold; fit all other transforms here."""
    available = np.isfinite(train).any(axis=0)
    if not available.any():
        raise ValueError("No input column is observed in this training fold.")
    imputer = SimpleImputer(strategy="median")
    train_filled = imputer.fit_transform(train[:, available])
    scaler = StandardScaler().fit(train_filled)
    test_filled = imputer.transform(test[:, available])
    train_z = torch.as_tensor(scaler.transform(train_filled), dtype=torch.float32)
    test_z = torch.as_tensor(scaler.transform(test_filled), dtype=torch.float32)
    return train_z, test_z, available


def fit_fold(raw, labels, descriptors, directions, train, test, variant, epochs, seed):
    torch.manual_seed(seed)
    x, x_test, _ = scaled_inputs(raw[train], raw[test])
    if not np.isfinite(labels[train]).any(axis=0).all():
        raise ValueError("A training fold has no labels for at least one of the three targets.")
    target_scaler = StandardScaler().fit(labels[train])
    y = torch.as_tensor(target_scaler.transform(labels[train]), dtype=torch.float32)

    z, z_test, observed, signs = None, None, None, None
    if variant != "mtnn":
        z, z_test, available = scaled_inputs(descriptors[train], descriptors[test])
        observed = torch.as_tensor(np.isfinite(descriptors[train][:, available]))
        signs = torch.as_tensor(directions[:, available], dtype=torch.float32)
    net = MultiTaskNet(x.shape[1], 0 if z is None else z.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z_step = z.detach().requires_grad_(True) if variant == "pi-mtnn" else z
        prediction = net(x, z_step)
        loss = observed_mse(prediction, y)
        if variant == "pi-mtnn":
            loss = loss + 0.002 * directional_penalty(prediction, z_step, observed, signs)
        loss.backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        scaled = net(x_test, z_test).numpy()
    return target_scaler.inverse_transform(scaled)


def cross_validate(raw, labels, descriptors, directions, variant, epochs, seed=DEFAULT_SEED, groups=None):
    """Return one held-out prediction per row. Group labels, when used, are supplied by the caller."""
    if variant not in {"mtnn", "mtnn-physics", "pi-mtnn"} or epochs < 1:
        raise ValueError("Specify a supported model and a positive number of epochs.")
    if groups is None:
        splits = KFold(n_splits=5, shuffle=True, random_state=seed).split(raw)
    else:
        if len(groups) != len(raw) or len(set(groups)) < 5:
            raise ValueError("Source-grouped five-fold CV requires one group per row and at least five sources.")
        splits = GroupKFold(n_splits=5).split(raw, groups=groups)
    predictions = np.full(labels.shape, np.nan)
    folds = np.zeros(len(raw), dtype=int)
    for fold, (train, test) in enumerate(splits, start=1):
        predictions[test] = fit_fold(raw, labels, descriptors, directions, train, test, variant, epochs, seed + fold)
        folds[test] = fold
    if not np.isfinite(predictions).all() or (folds == 0).any():
        raise RuntimeError("Incomplete or non-finite held-out predictions.")
    return folds, predictions


def fold_metrics(observed, predicted, folds):
    """Average the physical-unit RMSE and R² separately across defined validation folds."""
    summaries = []
    for j in range(len(TARGETS)):
        rmse, r2 = [], []
        for fold in np.unique(folds):
            mask = (folds == fold) & np.isfinite(observed[:, j])
            y = observed[mask, j]
            if not y.size:
                continue
            residual = y - predicted[mask, j]
            rmse.append(float(np.sqrt(np.mean(residual ** 2))))
            total = np.sum((y - y.mean()) ** 2)
            if y.size > 1 and total > 0:
                r2.append(float(1 - np.sum(residual ** 2) / total))
        summaries.append((np.mean(rmse) if rmse else np.nan, len(rmse),
                          np.mean(r2) if r2 else np.nan, len(r2)))
    return summaries


def read_groups(path, ids):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        if header.count("record_id") != 1 or header.count("source_id") != 1:
            raise ValueError("Group CSV needs one record_id column and one source_id column.")
        by_id = {}
        for row in reader:
            if None in row or not (row["record_id"] or "").strip():
                raise ValueError(f"Malformed group CSV at line {reader.line_num}.")
            key = int(row["record_id"])
            source = (row["source_id"] or "").strip()
            if key in by_id or not source:
                raise ValueError(f"Duplicate ID or missing source for record {key}.")
            by_id[key] = source
    if set(ids) != set(by_id):
        raise ValueError("Source IDs must cover exactly the records in the input CSV.")
    return [by_id[key] for key in ids]


def export_oof(path, ids, folds, labels, predictions, *, run_info=None):
    """Write values in physical units and carry the run description with each exported row."""
    path = Path(path)
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if len(folds) != len(ids) or labels.shape != (len(ids), len(TARGETS)) or predictions.shape != labels.shape:
        raise ValueError("Record IDs, folds and three-target predictions must align.")
    info = {} if run_info is None else run_info
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", encoding="utf-8", newline="")
    try:
        with stream:
            writer = csv.writer(stream)
            writer.writerow(["record_id", "fold", *["observed_" + t for t in TARGETS],
                             *["predicted_" + t for t in TARGETS], *info.keys()])
            for key, fold, obs, pred in zip(ids, folds, labels, predictions):
                writer.writerow([key, fold, *[v if np.isfinite(v) else "" for v in obs], *pred, *info.values()])
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--model", choices=["mtnn", "mtnn-physics", "pi-mtnn"], default="pi-mtnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--source-groups", type=Path, help="CSV of verified record_id,source_id pairs.")
    parser.add_argument("--export-oof", type=Path, help="Write held-out predictions in physical units to a new CSV file.")
    args = parser.parse_args()
    rows = read_records(args.data)
    digest = record_digest(rows)
    input_source = "Supplementary Table S4" if digest == TABLE_S4_DIGEST else "user-supplied records"
    ids = [row["record_id"] for row in rows]
    raw = np.asarray([[row[col] for col in FEATURES] for row in rows], dtype=float)
    labels = np.asarray([[row[col] for col in TARGETS] for row in rows], dtype=float)

    # At fixed resistivity the classical eddy-current term scales with gauge squared.
    gauge = raw[:, FEATURES.index("sheet_thickness_mm")]
    descriptors = (gauge ** 2)[:, None]
    directions = np.zeros((len(TARGETS), 1))
    directions[TARGETS.index("p1_5_50_w_kg"), 0] = +1
    groups = read_groups(args.source_groups, ids) if args.source_groups else None

    torch.set_num_threads(1)
    folds, predictions = cross_validate(raw, labels, descriptors, directions, args.model, args.epochs, groups=groups)
    protocol = "source-grouped" if groups is not None else "record-random"
    print(f"{input_source} | {len(ids)} records | {args.model} | {protocol} five-fold CV")
    print(f"First input row, record {ids[0]}, held out in fold {folds[0]}")
    print("Target and unit             Input value     Held-out prediction")
    names_with_units = [f"{name} ({unit})" for name, unit in zip(TARGETS, TARGET_UNITS)]
    for j, name in enumerate(names_with_units):
        value = f"{labels[0, j]:.6g}" if np.isfinite(labels[0, j]) else "missing"
        print(f"{name:26} {value:>12} {predictions[0, j]:23.6g}")
    print("\nMean metrics across validation folds")
    print("Target and RMSE unit        Mean fold RMSE (folds)     Mean fold R2 (folds)")
    for name, (rmse, n_rmse, r2, n_r2) in zip(names_with_units, fold_metrics(labels, predictions, folds)):
        print(f"{name:26} {rmse:12.5g} ({n_rmse})       {r2:12.5g} ({n_r2})")
    if args.export_oof:
        run_info = {
            "run_scope": "five_fold_cross_validation",
            "input_source": input_source,
            "input_file": args.data.name,
            "records_sha256": digest,
            "model_variant": args.model,
            "split_protocol": protocol,
            "base_seed": DEFAULT_SEED,
            "epochs": args.epochs,
        }
        export_oof(args.export_oof, ids, folds, labels, predictions, run_info=run_info)
        print(f"Wrote {args.export_oof}")


if __name__ == "__main__":
    main()
