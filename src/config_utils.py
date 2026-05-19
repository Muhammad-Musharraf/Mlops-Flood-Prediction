import yaml
import os
from logger import logging

def load_config(config_path='config.yaml', params_path='params.yaml'):
    """Loads configuration from config.yaml and params.yaml."""
    config = {}
    params = {}

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logging.info(f"Warning: {config_path} not found.")

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
    else:
        logging.info(f"Warning: {params_path} not found.")
            
    return config, params

# Example Usage (for testing/demonstration)
if __name__ == "__main__":
    config, params = load_config()
    logging.info("Loaded Config:", config)
    logging.info("Loaded Params:", params)