import os
import sys
import yaml
import datetime
import logging

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
            validate_config(config_dict)
        except yaml.YAMLError as exc:
            logging.error(exc)

    logging.info("Config File Loaded")
    return config_dict


def validate_config(config_dict):
    logger.info("Validating Config File")
    return True


def get_logger(log_folder):
    os.makedirs(log_folder, exist_ok=True)
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d-%H.%M.%S")

    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(log_folder, f"logs-{timestamp}.txt"),
        filemode="w",
        format="%(asctime)-8s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s",
    )
    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
