import os
import logging
import argparse

from utils import load_config, get_logger, ValidateConfig, set_seed
from core.bow import bow
from core.ngram import ngram
from core.tfidf import tfidf
from core.hmm import hmm
from core.word2vec import word2vec
from core.glove import glove
from core.rnn import rnn
from core.lstm import lstm
from core.gru import gru
from core.seq2seq import seq2seq
from core.transformer import transformer
from core.bert import bert
from core.gpt import gpt

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# List of Parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "-A",
    "--algo",
    type=str,
    required=True,
    help="Algorithm to Test",
    choices=[
        "BOW",
        "NGRAM",
        "TFIDF",
        "HMM",
        "WORD2VEC",
        "GLOVE",
        "RNN",
        "LSTM",
        "SEQ2SEQ",
        "GRU",
        "TRANSFORMER",
        "BERT",
        "GPT",
    ],
)
parser.add_argument(
    "-C", "--config_path", type=str, required=True, help="Path to Config File"
)
parser.add_argument(
    "-L", "--log_folder", type=str, required=True, help="Path to Log Folder"
)
parser.add_argument(
    "-S", "--seed", type=int, default=2024, required=False, help="Seed to Reproduce the results"
)

args = parser.parse_args()


def main():
    # Initializing Log Folder
    log_folder = args.log_folder
    get_logger(log_folder)

    # Setting seed across libraries to reproduce results
    seed = args.seed
    set_seed(seed)

    algo = args.algo
    config_path = args.config_path

    # Loading config into a dictionary
    config_dict = load_config(config_path)
    validate_config = ValidateConfig(config_dict, algo)
    validate_config.run_verify()

    # Running Algorithm Training and Inference
    if algo == "BOW":
        algo = bow.BOW(config_dict)
        algo.run()
    elif algo == "NGRAM":
        algo = ngram.NGRAM(config_dict)
        algo.run()
    elif algo == "TFIDF":
        algo = tfidf.TFIDF(config_dict)
        algo.run()
    elif algo == "HMM":
        algo = hmm.HMM(config_dict)
        algo.run()
    elif algo == "WORD2VEC":
        algo = word2vec.Word2Vec(config_dict)
        algo.run()
    elif algo == "GLOVE":
        algo = glove.GloVe(config_dict)
        algo.run()
    elif algo == "RNN":
        algo = rnn.RNN(config_dict)
        algo.run()
    elif algo == "LSTM":
        algo = lstm.LSTM(config_dict)
        algo.run()
    elif algo == "GRU":
        algo = gru.GRU(config_dict)
        algo.run()
    elif algo == "SEQ2SEQ":
        algo = seq2seq.Seq2Seq(config_dict)
        algo.run()
    elif algo == "TRANSFORMER":
        algo = transformer.Transformer(config_dict)
        algo.run()
    elif algo == "BERT":
        algo = bert.BERT(config_dict)
        algo.run()
    elif algo == "GPT":
        algo = gpt.GPT(config_dict)
        algo.run()


if __name__ == "__main__":
    main()
