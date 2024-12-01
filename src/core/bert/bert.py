import os
import json
import scipy.special
import torch
import scipy
import logging
import numpy as np
import pandas as pd
import torch.nn as nn

from .dataset_pretrain import create_dataloader_pretrain, PreprocessBERTPretrain
from .dataset_finetune import create_data_loader_finetune, PreprocessBERTFinetune
from .model import BERTPretrainModel, BERTFinetuneModel
from .pretrain import BERTPretrainTrainer
from .finetune import BERTFinetuneTrainer
from plot_utils import plot_embed, plot_history


class BERT:
    """
    A class to run BERT data preprocessing, training and inference

    :param config_dict: Config Params Dictionary
    :type config_dict: dict
    """
    def __init__(self, config_dict):
        self.logger = logging.getLogger(__name__)
        self.config_dict = config_dict

    def run(self):
        """
        Runs BERT pretrain and finetune stages and saves output
        """
        self.trainer_pretrain, self.pretrain_history = self.run_pretrain()
        self.model_pretrain = self.trainer_pretrain.model

        self.trainer_finetune, self.finetune_history = self.run_finetune()
        self.model_finetune = self.trainer_finetune.model

        self.save_output()

    def run_pretrain(self):
        """
        Pretraining stage of BERT

        :return: BERT Pretrain Trainer and Training History
        :rtype: tuple (torch.nn.Module, dict)
        """
        val_split = self.config_dict["dataset"]["val_split"]
        test_split = self.config_dict["dataset"]["test_split"]
        batch_size = self.config_dict["dataset"]["batch_size"]
        seed = self.config_dict["dataset"]["seed"]

        self.bert_pretrain_ds = PreprocessBERTPretrain(self.config_dict)
        text_tokens, nsp_labels = self.bert_pretrain_ds.get_data()
        self.word2id = self.bert_pretrain_ds.word2id
        self.wordpiece = self.bert_pretrain_ds.wordpiece

        train_loader, val_loader, self.test_loader_pretrain = (
            create_dataloader_pretrain(
                text_tokens,
                nsp_labels,
                self.config_dict,
                self.word2id,
                val_split,
                test_split,
                batch_size,
                seed,
            )
        )

        model = BERTPretrainModel(self.config_dict)
        lr = self.config_dict["train"]["lr"]
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = BERTPretrainTrainer(model, optim, self.config_dict)

        self.logger.info(f"-----------BERT Pretraining-----------")
        history = trainer.fit(train_loader, val_loader)

        return trainer, history

    def run_finetune(self):
        """
        Finetuning stage of BERT

        :return: BERT Fientune Trainer and Training History
        :rtype: tuple (torch.nn.Module, dict)
        """
        val_split = self.config_dict["finetune"]["dataset"]["val_split"]
        test_split = self.config_dict["finetune"]["dataset"]["test_split"]
        batch_size = self.config_dict["finetune"]["dataset"]["batch_size"]
        seed = self.config_dict["finetune"]["dataset"]["seed"]

        self.bert_finetune_ds = PreprocessBERTFinetune(
            self.config_dict, self.wordpiece, self.word2id
        )
        tokens, start_ids, end_ids, topics = self.bert_finetune_ds.get_data()

        train_loader, val_loader, self.test_loader = create_data_loader_finetune(
            tokens,
            start_ids,
            end_ids,
            topics,
            val_split=val_split,
            test_split=test_split,
            batch_size=batch_size,
            seed=seed,
        )

        model = BERTFinetuneModel(self.config_dict)
        model = self.load_pretrain_weights(self.model_pretrain, model)

        lr = self.config_dict["finetune"]["train"]["lr"]
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = BERTFinetuneTrainer(model, optim, self.config_dict)

        self.logger.info(f"-----------BERT Finetuning-----------")
        history = trainer.fit(train_loader, val_loader)

        return trainer, history

    def run_infer_finetune(self):
        """
        Runs inference using Finetuned BERT

        :return: True and Predicted start, end ids
        :rtype: tuple (numpy.ndarray [num_samples,], numpy.ndarray [num_samples,], numpy.ndarray [num_samples,], numpy.ndarray [num_samples,])
        """
        start_ids, end_ids, enc_outputs = self.trainer_finetune.predict(
            self.test_loader
        )
        num_samples = len(start_ids)
        seq_len = self.config_dict["dataset"]["seq_len"]

        start = self.model_finetune.start.cpu().detach().numpy()
        end = self.model_finetune.end.cpu().detach().numpy()

        cxt_enc_output = enc_outputs[:, seq_len // 2 :, :]
        start_muls = np.matmul(cxt_enc_output, start).squeeze()
        start_probs = scipy.special.softmax(start_muls, axis=1)
        end_muls = np.matmul(cxt_enc_output, end).squeeze()

        start_ids_pred = np.argmax(start_probs, axis=1)
        start_max_muls = start_muls[np.arange(num_samples), start_ids_pred]
        sum_muls = start_max_muls.reshape(-1, 1) + end_muls

        def get_max_end_index(arr, start_indices):
            num_rows, num_cols = arr.shape
            mask = np.stack([np.arange(num_cols)] * num_rows) <= np.expand_dims(
                start_indices, 1
            )
            masked_arr = arr
            masked_arr[mask] = -1e9
            max_indices = np.argmax(masked_arr, axis=1)

            return max_indices

        end_ids_pred = get_max_end_index(sum_muls, start_ids_pred)

        return start_ids, end_ids, start_ids_pred, end_ids_pred

    def load_pretrain_weights(self, pretrain_model, finetune_model):
        """
        Copies pretrain weights to finetune BERT model object

        :param pretrain_model: Pretrain BERT model
        :type pretrain_model: torch.nn.Module
        :param finetune_model: Finetune BERT model 
        :type finetune_model: torch.nn.Module
        :return: Finetune BERT model with Pretrained weights
        :rtype: torch.nn.Module
        """
        for i, layer in enumerate(pretrain_model.encoder_layers):
            finetune_model.encoder_layers[i].load_state_dict(layer.state_dict())

        finetune_model.embed_layer.load_state_dict(
            pretrain_model.embed_layer.state_dict()
        )

        return finetune_model

    def save_output(self):
        """
        Saves Training and Inference results
        """
        output_folder = self.config_dict["paths"]["output_folder"]

        self.logger.info(f"Saving Outputs {output_folder}")
        with open(os.path.join(output_folder, "pretraining_history.json"), "w") as fp:
            json.dump(self.pretrain_history, fp)
        plot_history(self.pretrain_history, output_folder, "Pretrain History")

        with open(os.path.join(output_folder, "finetuning_history.json"), "w") as fp:
            json.dump(self.finetune_history, fp)
        plot_history(self.finetune_history, output_folder, "Finetune History")

        embeds = self.model_pretrain.embed_layer.weight.detach().numpy()
        vocab = list(self.bert_pretrain_ds.word2id.keys())
        plot_embed(embeds, vocab, output_folder, fname="Tokens Embeddings TSNE")

        start_ids, end_ids, start_ids_pred, end_ids_pred = self.run_infer_finetune()
        df = pd.DataFrame.from_dict(
            {
                "Start ID": start_ids,
                "End ID": end_ids,
                "Start ID Pred": start_ids_pred,
                "End ID Pred": end_ids_pred,
            }
        )
        df.to_csv(os.path.join(output_folder, "Finetune Predictions.csv"), index=False)
