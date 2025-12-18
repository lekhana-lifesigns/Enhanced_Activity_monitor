# pipeline/utils/model_registry.py
"""
Model Memory Sharing Registry.
Prevents duplicate model loading and reduces memory usage.
"""

import logging
from typing import Dict, Optional, Any
import threading

log = logging.getLogger("model_registry")


class ModelRegistry:
    """
    Shared model registry to prevent duplicate model loading.
    Thread-safe singleton pattern.
    """
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_model(cls, model_type: str, model_path: Optional[str] = None, 
                  loader_func=None, *args, **kwargs) -> Any:
        """
        Get or load a model from the registry.
        
        Args:
            model_type: Type of model (e.g., "yolo", "pose", "temporal")
            model_path: Path to model file (used as part of key)
            loader_func: Function to load the model if not in registry
            *args, **kwargs: Arguments to pass to loader_func
        
        Returns:
            Loaded model instance
        """
        # Create unique key
        key = f"{model_type}_{model_path}" if model_path else model_type
        
        with cls._lock:
            if key not in cls._models:
                if loader_func is None:
                    log.warning("Model %s not in registry and no loader provided", key)
                    return None
                
                log.info("Loading model: %s", key)
                try:
                    cls._models[key] = loader_func(*args, **kwargs)
                    log.info("Model loaded successfully: %s", key)
                except Exception as e:
                    log.error("Failed to load model %s: %s", key, e)
                    return None
            
            return cls._models[key]
    
    @classmethod
    def register_model(cls, model_type: str, model_path: Optional[str], model_instance: Any):
        """Manually register a model instance."""
        key = f"{model_type}_{model_path}" if model_path else model_type
        with cls._lock:
            cls._models[key] = model_instance
            log.info("Model registered: %s", key)
    
    @classmethod
    def clear_registry(cls):
        """Clear all models from registry (use with caution)."""
        with cls._lock:
            cls._models.clear()
            log.info("Model registry cleared")
    
    @classmethod
    def get_registry_size(cls) -> int:
        """Get number of models in registry."""
        with cls._lock:
            return len(cls._models)

