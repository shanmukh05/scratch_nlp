import os
import sys
import yaml
import torch
import random
import datetime
import logging
import numpy as np

from configs import configDictDType, MainKeysDict

logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Loading YAML Config file as a Dictionary 

    :param config_path: Path to Config File
    :type config_path: str
    :return: Config Params Dictionary
    :rtype: dict
    """
    with open(config_path, "r") as stream:
        try:
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)

    logging.info("Config File Loaded")
    return config_dict

def set_seed(seed):
    """
    Setting seed across Libraries to reproduce results

    :param seed: Seed value
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ValidateConfig:
    """
    Validating Config File

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    :param algo: Name of the Algorithm
    :type algo: str
    """

    def __init__(self, config_dict, algo):
        self.config_dict = config_dict
        self.algo = algo

    def run_verify(self):
        """
        Config Params Keys and Values Verification 
        """
        logger.info("Validating Config File")
        self.verify_main_keys(self.config_dict.keys())
        self.verify_values()

    def check_float(self, key, val):
        """
        To check whether given key whose value is float has a valid value or not

        :param key: Param Key
        :type key: str
        :param val: Param value
        :type val: float
        """        
        pass

    def check_int(self, key, val):
        """
        To check whether given key whose value is int has a valid value or not

        :param key: Param Key
        :type key: str
        :param val: Param value
        :type val: int
        """       
        pass

    def check_string(self, key, val):
        """
        To check whether given key whose value is str has a valid value or not

        :param key: Param Key
        :type key: str
        :param val: Param value
        :type val: str
        """       
        pass

    def check_paths(self, key, val):
        """
        To check whether given key whose value is a filepath has a valid value or not

        :param key: Param Key
        :type key: str
        :param val: Param value
        :type val: str
        """       
        pass

    def check_list(self, key, val):
        """
        To check whether given key whose value is list has a valid value or not

        :param key: Param Key
        :type key: str
        :param val: Param value
        :type val: list
        """  
        pass

    def compare_dtype(self, key, val):
        """
        To check whether given key whose value has a valid dtype or not

        :param key: Param Key
        :type key: str
        :param val: Param value
        :type val: float/int/str/list
        """      
        type_abs = configDictDType[key]
        type_cfg = type(val)

        if type_abs != type_cfg:
            logging.error(f"Dtype of {key} should be {type_abs}")

    def verify_values(self):
        """
        Verifying the Datatypes of all the Parameters in Config
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
        Verifying whether Config has all the required keys or not

        :param keys: Parent Config Parameters
        :type keys: list
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
    Initializing Log File

    :param log_folder: Path to folder where Log file is added
    :type log_folder: str
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
