from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import requests

from data_loader import ALL_DATASETS, DATA_DIR, get_dataset_paths, load_data, prepare_train_data, prepare_test_samples

NASA_CMAPSS_ZIP_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
NASA_PCOE_ZIP_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
DEFAULT_DATASETS = ALL_DATASETS
EXPECTED = {
    "FD001": (20631, 13096, 100, 100),
    "FD002": (53759, 33991, 260, 259),
    "FD003": (24720, 16596, 100, 100),
    "FD004": (61249, 41214, 249, 248),
}


def download_zip(destination: Path, url: str = NASA_CMAPSS_ZIP_URL) -> None:
    pending = destination.with_suffix(".part")
    try:
        with requests.get(url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            with pending.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    handle.write(chunk)
        if not zipfile.is_zipfile(pending):
            raise ValueError("NASA download was not a complete ZIP archive")
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)


def extract_datasets(zip_path: Path, destination: Path, datasets: tuple[str, ...]) -> None:
    names = {f"train_{dataset}.txt" for dataset in datasets}
    names.update({f"test_{dataset}.txt" for dataset in datasets})
    names.update({f"RUL_{dataset}.txt" for dataset in datasets})
    with zipfile.ZipFile(zip_path) as archive:
        nested = next((name for name in archive.namelist() if Path(name).name == "CMAPSSData.zip"), None)
        if nested:
            with zipfile.ZipFile(io.BytesIO(archive.read(nested))) as inner:
                files = read_dataset_members(inner, names)
        else:
            files = read_dataset_members(archive, names)
    for name, content in files.items():
        path = destination / name
        pending = path.with_suffix(".part")
        pending.write_bytes(content)
        pending.replace(path)


def read_dataset_members(archive: zipfile.ZipFile, names: set[str]) -> dict[str, bytes]:
    members = {Path(name).name: name for name in archive.namelist() if not name.startswith("__MACOSX/")}
    if missing := names - members.keys():
        raise ValueError(f"NASA archive is missing {sorted(missing)}")
    names = names | ({"readme.txt"} & members.keys())
    return {name: archive.read(members[name]) for name in sorted(names)}


def audit_dataset(dataset: str) -> dict:
    train, test, rul = load_data(dataset)
    dimensions = (len(train), len(test), train.unit_id.nunique(), test.unit_id.nunique())
    if dimensions != EXPECTED[dataset] or len(rul) != dimensions[-1]:
        raise ValueError(f"{dataset} has unexpected dimensions {dimensions}, RUL rows {len(rul)}")
    for frame in (train, test):
        ordered = frame.sort_values(["unit_id", "cycle"])
        if ordered.duplicated(["unit_id", "cycle"]).any() or not np.array_equal(
            ordered["cycle"].to_numpy(), ordered.groupby("unit_id").cumcount().to_numpy() + 1
        ):
            raise ValueError(f"{dataset} has duplicate or nonconsecutive engine cycles")
    if (rul["RUL"] < 0).any() or (rul["RUL"] % 1 != 0).any():
        raise ValueError(f"{dataset} has invalid terminal RUL")
    prepared, features, _ = prepare_train_data(train, max_rul=None)
    _, labels, ids = prepare_test_samples(test, rul, features, max_rul=None)
    terminal_train = prepared.groupby("unit_id")["RUL"].min()
    if (terminal_train != 0).any() or not np.array_equal(labels, rul["RUL"].to_numpy()[ids - 1]):
        raise ValueError(f"{dataset} has misaligned RUL labels")
    return {
        "train_rows": dimensions[0], "test_rows": dimensions[1], "columns": len(train.columns),
        "train_engines": dimensions[2], "test_engines": dimensions[3], "rul_rows": len(rul),
        "missing_values": 0, "duplicate_engine_cycles": 0,
        "test_rul_above_125": int((rul["RUL"] > 125).sum()),
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in get_dataset_paths(dataset).values()},
        "train_feature_distributions": train[features].describe().round(8).to_dict(),
        "test_feature_distributions": test[features].describe().round(8).to_dict(),
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "CMAPSSData.zip"
    if not zipfile.is_zipfile(zip_path):
        print(f"Downloading NASA C-MAPSS from {NASA_CMAPSS_ZIP_URL}")
        try:
            download_zip(zip_path)
        except (requests.RequestException, ValueError) as exc:
            print(f"Open Data download failed ({type(exc).__name__}); trying NASA PCoE")
            download_zip(zip_path, NASA_PCOE_ZIP_URL)
    extract_datasets(zip_path, DATA_DIR, DEFAULT_DATASETS)
    report = {
        "archive_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
        "datasets": {dataset: audit_dataset(dataset) for dataset in DEFAULT_DATASETS},
    }
    (DATA_DIR / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for dataset, audit in report["datasets"].items():
        print(f"{dataset}: {audit['train_rows']:,}/{audit['test_rows']:,} train/test rows, "
              f"{audit['train_engines']}/{audit['test_engines']} engines, zero missing or duplicate cycles")


if __name__ == "__main__":
    main()
