import os
import sys
import argparse
import logging

from utils import load_config, get_logger, ValidateConfig
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

### TODO Set all seeds

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

args = parser.parse_args()


def main():
    log_folder = args.log_folder
    get_logger(log_folder)

    algo = args.algo
    config_path = args.config_path

    config_dict = load_config(config_path)
    validate_config = ValidateConfig(config_dict, algo)
    validate_config.run_verify()

    if algo == "BOW":
        algo = bow.BOW(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\bow.yaml" --algo "BOW" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\bow"
    elif algo == "NGRAM":
        algo = ngram.NGRAM(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\ngram.yaml" --algo "NGRAM" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\ngram"
    elif algo == "TFIDF":
        algo = tfidf.TFIDF(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\tfidf.yaml" --algo "TFIDF" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\tfidf"
    elif algo == "HMM":
        algo = hmm.HMM(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\hmm.yaml" --algo "HMM" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\hmm"
    elif algo == "WORD2VEC":
        algo = word2vec.Word2Vec(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\word2vec.yaml" --algo "WORD2VEC" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\word2vec"
    elif algo == "GLOVE":
        algo = glove.GloVe(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\glove.yaml" --algo "GLOVE" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\glove"
    elif algo == "RNN":
        algo = rnn.RNN(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\rnn.yaml" --algo "RNN" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\rnn"
    elif algo == "LSTM":
        algo = lstm.LSTM(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\lstm.yaml" --algo "LSTM" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\lstm"
    elif algo == "GRU":
        algo = gru.GRU(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\gru.yaml" --algo "GRU" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\gru"
    elif algo == "SEQ2SEQ":
        algo = seq2seq.Seq2Seq(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\seq2seq.yaml" --algo "SEQ2SEQ" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\seq2seq"
    elif algo == "TRANSFORMER":
        algo = transformer.Transformer(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\transformer.yaml" --algo "TRANSFORMER" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\transformer"
    elif algo == "BERT":
        algo = bert.BERT(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\bert.yaml" --algo "BERT" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\bert"
    elif algo == "GPT":
        algo = gpt.GPT(config_dict)
        algo.run()
        # python main.py --config_path "D:\Learning\NLP\Projects\scratch_nlp\configs\gpt.yaml" --algo "GPT" --log_folder "D:\Learning\NLP\Projects\scratch_nlp\output\gpt"


if __name__ == "__main__":
    main()
