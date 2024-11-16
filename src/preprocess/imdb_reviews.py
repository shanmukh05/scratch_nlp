import os
import glob
from tqdm import tqdm
import numpy as np
import logging

from .utils import preprocess_text


class PreprocessIMDB:
    """
    Loading and Generating Reviews, labels for IMDB dataset

    :param root_path: Root Folder with all the classes Folders with txt files for each sample or can have txt files
    :type root_path: str
    :param explore_folder: Whether the root_path has classes folder or txt files
    :type explore_folder: bool
    :param num_samples: How many samples to select from each folder
    :type num_samples: int
    :param operations: Any combinations of {'lcase', 'remalpha', 'stopwords', 'stemming'}. list of preprocessing Operations
    :type operations: list
    :param randomize: Select first num_samples or at random
    :type randomize: bool
    """
    def __init__(self, root_path, explore_folder, num_samples, operations, randomize):
        self.logger = logging.getLogger(__name__)
        self.extract_data(root_path, explore_folder, num_samples, randomize)
        self.logger.info("Extracted Data from TXT Files")
        self.operations = operations

    def run(self):
        """
        Preprocessing list of sentences
        """
        self.text_ls = [preprocess_text(i, self.operations) for i in self.text_ls]

    def extract_data(self, root_path, explore_folder, num_samples, randomize):
        """
        Extracting data from txt files

        :param root_path: Root Folder with all the classes Folders with txt files for each sample or can have txt files
        :type root_path: str
        :param explore_folder: Whether the root_path has classes folder or txt files
        :type explore_folder: bool
        :param num_samples: How many samples to select from each folder
        :type num_samples: int
        :param randomize: Select first num_samples or at random
        :type randomize: bool
        """
        self.text_ls, self.label_ls = [], []
        if explore_folder:
            folders = os.listdir(root_path)
            for folder in folders:
                fold_path = os.path.join(root_path, folder)
                text_ls = self.extract_data_folder(fold_path, num_samples, randomize)
                self.text_ls.extend(text_ls)
                self.label_ls.extend([folder] * num_samples)
        else:
            text_ls = self.extract_data_folder(root_path, num_samples, randomize)
            self.text_ls.extend(text_ls)
            folder = os.path.basename(root_path)
            self.label_ls.extend([folder] * num_samples)

    def extract_data_folder(self, fold_path, num_samples, randomize):
        """
        Extracting txt data from each folder

        :param fold_path: Path to Folder
        :type fold_path: str
        :param num_samples: How many samples to select from each folder
        :type num_samples: int
        :param randomize: Select first num_samples or at random
        :type randomize: bool
        :return: List of sentences from the folder
        :rtype: list
        """
        text_ls = []
        path_ls = glob.glob(f"{fold_path}/*.txt")
        if randomize:
            path_ids = np.random.choice(np.arange(len(path_ls)), size=num_samples)
        else:
            path_ids = np.arange(num_samples)

        desc = f"Extracting from: {os.path.basename(fold_path)}"
        for id in tqdm(path_ids, desc=desc):
            f = open(path_ls[id], "r", encoding="utf8", errors="ignore")
            text = " ".join(f.readlines())
            text_ls.append(text)
            f.close()
        return text_ls
