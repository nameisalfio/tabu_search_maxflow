"""
Configuration utilities
"""

import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    """Gestione configurazione progetto"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carica configurazione da YAML"""
        if not self.config_path.exists():
            print(f"⚠️  Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"✅ Config loaded from {self.config_path}")
                return config
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configurazione di default"""
        return {
            'tabu_search': {
                'max_iterations': 20000,
                'tabu_list_size': 10,
                'aspiration_enabled': True,
                'delta_step': 1
            },
            'experiments': {
                'num_runs': 10,
                'random_seeds': list(range(42, 52))
            },
            'visualization': {
                'theme': 'dark',
                'save_plots': True,
                'show_plots': False,
                'plot_format': 'png',
                'figsize': [12, 8]
            },
            'output': {
                'save_results': True,
                'verbose': True
            }
        }
    
    def get(self, key_path: str, default=None):
        """Ottiene valore con dot notation (es: 'tabu_search.max_iterations')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Imposta valore con dot notation"""
        keys = key_path.split('.')
        config_ref = self.config
        
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        config_ref[keys[-1]] = value
    
    def save(self):
        """Salva configurazione corrente"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            print(f"✅ Config saved to {self.config_path}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
