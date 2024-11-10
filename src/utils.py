import os
import sys
import yaml
import datetime
import logging

from configs import configDictDType, MainKeysDict

logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    _summary_

    :param config_path: _description_
    :type config_path: _type_
    :return: _description_
    :rtype: _type_
    """
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

    logging.info("Config File Loaded")
    return config_dict


class ValidateConfig:
    """
    _summary_

    :param config_dict: _description_
    :type config_dict: _type_
    :param algo: _description_
    :type algo: _type_
    """

    def __init__(self, config_dict, algo):
        self.config_dict = config_dict
        self.algo = algo

    def run_verify(self):
        """
        _summary_
        """
        logger.info("Validating Config File")
        self.verify_main_keys(self.config_dict.keys())
        self.verify_values()

    def check_float(self, key, val):
        pass

    def check_int(self, key, val):
        pass

    def check_string(self, key, val):
        pass

    def check_paths(self, key, val):
        pass

    def check_list(self, key, val):
        pass

    def compare_dtype(self, key, val):
        type_abs = configDictDType[key]
        type_cfg = type(val)

        if type_abs != type_cfg:
            logging.error(f"Dtype of {key} should be {type_abs}")

    def verify_values(self):
        """
        _summary_
        """
        for k, v in self.config_dict.items():
            if isinstance(v, dict):
                for k_, v_ in v.items():
                    if type(v_) is dict:
                        for k__, v__ in v_.items():
                            self.compare_dtype(k__, v__)
                    else:
                        self.compare_dtype(k_, v_)
            else:
                self.compare_dtype(k, v)

    def verify_main_keys(self, keys):
        """
        _summary_

        :param keys: _description_
        :type keys: _type_
        """
        for key in keys:
            true_val = MainKeysDict[self.algo][key]

            if isinstance(true_val, list):
                cfg_val = list(self.config_dict[key].keys())
                true_val.sort()
                cfg_val.sort()

                if true_val != cfg_val:
                    logging.error(f"Config Keys for {key} doesn't match Default Config")
            elif isinstance(true_val, dict):
                for k in self.config_dict[key].keys():
                    true_val_k = MainKeysDict[self.algo][key][k]
                    cfg_val_k = list(self.config_dict[key][k].keys())

                    true_val_k.sort()
                    cfg_val_k.sort()

                    if true_val_k != cfg_val_k:
                        logging.error(
                            f"Config Keys for {k} doesn't match Default Config"
                        )
            else:
                if not isinstance(self.config_dict[key], MainKeysDict[self.algo][key]):
                    logging.error(
                        f"Config Key {key} should be of type {MainKeysDict[self.algo][key]}"
                    )


def get_logger(log_folder):
    """
    _summary_

    :param log_folder: _description_
    :type log_folder: _type_
    """
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
