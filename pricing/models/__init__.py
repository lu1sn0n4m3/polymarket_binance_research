import importlib

from pricing.models.base import Model, CalibrationResult

# Registry of built-in models: name -> (module_path, class_name)
MODEL_REGISTRY = {
    "gaussian": ("pricing.models.gaussian", "GaussianModel"),
    "gaussian_t": ("pricing.models.gaussian_t", "GaussianTModel"),
}


def get_model(name: str) -> Model:
    """Instantiate a model by registry name or fully-qualified 'module.ClassName'."""
    if name in MODEL_REGISTRY:
        module_path, class_name = MODEL_REGISTRY[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()
    if "." in name:
        parts = name.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])()
    raise ValueError(
        f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
    )
