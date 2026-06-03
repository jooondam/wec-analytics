"""
Model persistence utilities for the wec_analytics ML layer.

Saves and loads fitted sklearn models using joblib, with a JSON sidecar
that records training provenance — sessions used, evaluation scores,
package versions, and save timestamp. The sidecar is plain text so model
metadata can be inspected without loading the binary artefact.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn


def save_model(model, filepath: str | Path, metadata: dict | None = None) -> Path:
    """
    Save a fitted sklearn model to disk with an optional JSON sidecar.

    The model is saved using joblib (preferred over pickle for sklearn
    pipelines containing large numpy arrays). If metadata is provided,
    a sidecar file with the same stem and a .json extension is written
    alongside the model file. The sidecar always includes a save timestamp
    and the sklearn version, regardless of what the caller passes in.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Any fitted sklearn Pipeline or estimator.
    filepath : str or Path
        Destination path for the model file, e.g.
        'models_trained/20240504_133000_pace_linear.joblib'.
        Parent directories are created automatically.
    metadata : dict, optional
        Arbitrary key-value pairs to record alongside the model — e.g.
        training session URLs, evaluation scores, feature column names,
        random seed. Merged with auto-generated fields before saving.

    Returns
    -------
    Path
        Resolved path to the saved model file.

    Examples
    --------
    >>> save_model(
    ...     fitted_pipeline,
    ...     "models_trained/20240504_pace.joblib",
    ...     metadata={
    ...         "training_sessions": ["201905041330_Race"],
    ...         "rmse_mean": 0.843,
    ...         "feature_columns": ["stint_age", "rolling_pace", "lap_number", "car_class"],
    ...         "random_seed": 42,
    ...     }
    ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # save the model binary
    joblib.dump(model, filepath)

    # build sidecar — start with auto-generated fields, then merge caller metadata
    # so that save_timestamp and sklearn_version are always present and never
    # accidentally overwritten by the caller
    sidecar = {
        "save_timestamp": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    if metadata:
        sidecar.update(metadata)

    sidecar_path = filepath.with_suffix(".json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Model saved: {filepath}")
    print(f"Metadata saved: {sidecar_path}")

    return filepath


def load_model(filepath: str | Path) -> tuple:
    """
    Load a fitted sklearn model and its metadata sidecar from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the joblib model file. The sidecar is expected at the same
        path with a .json extension. If the sidecar is missing, metadata
        is returned as an empty dict with a warning rather than raising —
        old model files without sidecars should still be loadable.

    Returns
    -------
    tuple[estimator, dict]
        The fitted sklearn model and its metadata dict. If no sidecar
        exists, the dict is empty.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No model file found at '{filepath}'. "
            "Check the path or run save_model first."
        )

    model = joblib.load(filepath)

    # sidecar is optional — a missing sidecar is a warning, not a crash,
    # so that models saved before the sidecar convention was introduced
    # remain loadable
    sidecar_path = filepath.with_suffix(".json")
    if sidecar_path.exists():
        with open(sidecar_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        print(f"Warning: no metadata sidecar found at '{sidecar_path}'. Returning empty dict.")
        metadata = {}

    return model, metadata


def make_versioned_path(
    name: str,
    directory: str | Path = "models_trained",
    suffix: str = ".joblib",
) -> Path:
    """
    Build a timestamped model filepath to avoid accidental overwrites.

    Parameters
    ----------
    name : str
        Descriptive model name, e.g. 'pace_linear' or 'pace_gradient_boost'.
    directory : str or Path, optional
        Directory to save into. Default 'models_trained'.
    suffix : str, optional
        File extension. Default '.joblib'.

    Returns
    -------
    Path
        e.g. Path('models_trained/20240504_133012_pace_linear.joblib')

    Examples
    --------
    >>> path = make_versioned_path("pace_linear")
    >>> save_model(fitted_pipeline, path, metadata={...})
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{name}{suffix}"
    return Path(directory) / filename