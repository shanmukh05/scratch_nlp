configDictDType = {
    ## paths
    "input_file": str,
    "input_folder": str,
    "output_folder": str,
    "test_folder": str,
    "image_folder": str,
    "captions_file": str,
    "pretrain_weights": str,
    "test_file": str,

    ## dataset
    "num_samples": int,
    "explore_folder": bool,
    "test_size": float,
    "seed": int,
    "num_vocab": int,
    "context": int,
    "val_split": float,
    "batch_size": int,
    "seq_len": int,
    "num_classes": int,
    "test_samples": int,
    "train_samples": int,
    "num_extra_tokens": int,
    "test_split": float,
    "num_src_vocab": int,
    "num_tgt_vocab": int,
    "train_corpus": list,
    "test_corpus": list,
    "num_topics": int,
    "num_sents_per_doc": int,
    "random_lines": bool,

    ## model
    "output_label": bool,
    "tf_mode": str,
    "idf_mode": str,
    "embed_dim": int,
    "device": str,
    "num_layers": int,
    "h_dim": list,
    "x_dim": list,
    "clf_dim": list,
    "image_backbone": str,
    "encoder_h_dim": list,
    "encoder_x_dim": list,
    "decoder_y_dim": list,
    "d_model": int,
    "d_ff": int,
    "num_heads": int,
    "dropout": float,

    ## train
    "epochs": int,
    "lr": float,
    "x_max": int,
    "alpha": float,
    "eval_metric": str,
    "bleu_n": int,
    "rouge_n_n": int,
    "rouge_s_n": int,

    ## test
    "predict_tokens": int,

    ## preprocess
    "operations": list,
    "randomize": bool,
    "image_dim": list,
    "prediction": float,
    "mask": float,
    "random": float,
    "next": float,

    ## visualize
    "visualize": bool,

}

MainKeysDict = {
    "BOW": {
        "paths": ["input_folder", "output_folder"],
        "dataset": ["num_samples", "explore_folder"],
        "model": ["output_label"],
        "preprocess": ["operations", "randomize"],
        "visualize": bool
    },
    "NGRAM": {
        "paths": ["input_folder", "output_folder"],
        "dataset": ["num_samples", "explore_folder"],
        "preprocess": ["operations", "randomize"],
        "visualize": bool
    },
    "TFIDF": {
        "paths": ["input_folder", "output_folder"],
        "dataset": ["num_samples", "explore_folder"],
        "model": ["tf_mode", "idf_mode", "output_label"],
        "preprocess": ["operations", "randomize"],
        "visualize": bool
    },
    "HMM": {
        "paths": ["input_folder", "output_folder"],
        "dataset": ["num_samples", "test_size", "seed"],
        "model": ["output_label"],
        "visualize": bool
    },
    "WORD2VEC": {
        "paths": ["input_folder", "output_folder"],
        "dataset": ["num_samples", "explore_folder", "num_vocab", "context", "seed",  "val_split", "batch_size"],
        "preprocess": ["operations", "randomize"],
        "model": ["embed_dim", "device"],
        "train": ["epochs", "lr"],
        "visualize": bool
    },
    "GLOVE": {
        "paths": ["input_folder", "output_folder"],
        "dataset": ["num_samples", "explore_folder", "num_vocab", "context", "seed",  "val_split", "batch_size"],
        "preprocess": ["operations", "randomize"],
        "model": ["embed_dim", "device"],
        "train": ["epochs", "lr", "x_max", "alpha"],
        "visualize": bool
    },
    "RNN": {
        "paths": ["input_folder", "test_folder", "output_folder"],
        "dataset": ["num_samples", "test_samples", "explore_folder", "num_vocab", "seed", "val_split", "batch_size", "seq_len", "num_classes"],
        "preprocess": ["operations", "randomize"],
        "model": ["embed_dim", "num_layers", "h_dim", "x_dim", "clf_dim", "device"],
        "train": ["epochs", "lr", "eval_metric"],
        "visualize": bool
    },
    "LSTM": {
        "paths": ["image_folder", "captions_file", "output_folder"],
        "dataset": ["train_samples", "test_samples", "val_split", "num_vocab", "num_extra_tokens", "seq_len", "seed", "batch_size"],
        "preprocess": ["operations", "randomize", "image_dim"],
        "model": ["embed_dim", "num_layers", "h_dim", "x_dim", "device", "image_backbone"],
        "train": ["epochs", "lr", "eval_metric", "bleu_n", "rouge_n_n", "rouge_s_n"],
        "visualize": bool
    },
    "SEQ2SEQ": {
        "paths": ["input_file", "output_folder"],
        "dataset": ["num_samples", "val_split", "test_split", "num_src_vocab", "num_tgt_vocab", "seq_len", "seed", "batch_size"],
        "preprocess": ["operations", "randomize"],
        "model": ["embed_dim", "num_layers", "encoder_x_dim", "encoder_h_dim", "decoder_y_dim", "device"],
        "train": ["epochs", "lr", "eval_metric", "bleu_n", "rouge_n_n", "rouge_s_n"],
        "visualize": bool
    },
    "GRU": {
        "paths": ["output_folder"],
        "dataset": ["train_corpus", "train_samples", "val_split", "test_corpus", "test_samples", "num_vocab", "seq_len", "seed", "batch_size"],
        "preprocess": ["operations", "randomize"],
        "model": ["embed_dim", "h_dim", "x_dim", "device"],
        "train": ["epochs", "lr", "eval_metric"],
        "visualize": bool
    },
    "TRANSFORMER": {
        "paths": ["input_file", "output_folder"],
        "dataset": ["num_samples", "val_split", "test_split", "num_vocab", "num_extra_tokens", "seq_len", "seed", "batch_size"],
        "preprocess": ["randomize", "operations"],
        "model": ["d_model", "d_ff", "num_heads", "num_layers", "dropout"],
        "train": ["epochs", "lr", "eval_metric", "bleu_n", "rouge_n_n", "rouge_s_n"],
        "visualize": bool
    },
    "BERT": {
        "paths": ["input_file", "output_folder"],
        "dataset": ["num_samples", "val_split", "test_split", "num_vocab", "num_extra_tokens", "seq_len", "seed", "batch_size"],
        "preprocess": ["randomize", "operations", "replace_token", "sentence_pair"],
        "model": ["d_model", "d_ff", "num_heads", "num_layers", "dropout"],
        "train": ["epochs", "lr"],
        "visualize": bool,
        "finetune": {
            "paths": ["input_file", "pretrain_weights"],
            "dataset": ["num_topics", "val_split", "test_split", "seed", "batch_size"],
            "preprocess": ["randomize", "operations"],
            "train": ["epochs", "lr"]
        },
        "visualize": bool,
        "replace_token": ["prediction", "mask", "random"],
        "sentence_pair": ["next"],
    },
    "GPT": {
        "paths": ["input_folder", "test_file", "output_folder"],
        "dataset": ["num_sents_per_doc", "random_lines", "val_split", "test_samples", "num_vocab", "num_extra_tokens", "seq_len", "seed", "batch_size"],
        "preprocess": ["randomize", "operations"],
        "model": ["d_model", "d_ff", "num_heads", "num_layers", "dropout"],
        "train": ["epochs", "lr", "eval_metric", "bleu_n", "rouge_n_n", "rouge_s_n"],
        "test": ["predict_tokens"],
        "visualize": bool,
    }
}