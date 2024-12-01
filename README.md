![Logo](https://raw.githubusercontent.com/shanmukh05/scratch_nlp/main/assets/logo.png)

[![PyPI version](https://badge.fury.io/py/yourlibraryname.svg)](https://badge.fury.io/py/yourlibraryname)
[![Build Status](https://travis-ci.org/yourusername/yourlibraryname.svg?branch=master)](https://travis-ci.org/yourusername/yourlibraryname)
[![Coverage Status](https://coveralls.io/repos/github/yourusername/yourlibraryname/badge.svg?branch=master)](https://coveralls.io/github/yourusername/yourlibraryname?branch=master)

# Scratch NLP 🧠

Library with foundational NLP Algorithms implemented from scratch using PyTorch.

## Table of Contents 📋

- [Documentation](#documentation-📝)
- [Installation](#installation-⬇️)
- [Features](#features-🛠️)
- [Examples](#examples-🌟)
- [Contributing](#contributing-🤝)
- [Acknowledgements](#acknowledgements-💡)
- [About Me](#about-me-👤)
- [Lessons Learned](#lessons-learned-📌)
- [License](#license-⚖️)
- [Feedback](#feedback-📣)


## Documentation 📝

[Documentation](https://shanmukh05.github.io/scratch_nlp/)


## Installation ⬇️

### Install using pip

```bash
   pip install scratch-nlp
```
    
### Install Manually for development

Clone the repo

```bash
  gh repo clone shanmukh05/scratch_nlp
```

Install dependencies

```bash
  pip install -r requirements.txt
```


## Features 🛠️

- Algorithms
  - Bag of Words
  - Ngram
  - TF-IDF
  - Hidden Markov Model
  - Word2Vec
  - GloVe
  - RNN (Many to One)
  - LSTM (One to Many)
  - GRU (Many to Many Synced)
  - Seq2Seq + Attention (Many to Many)
  - Transformer
  - BERT
  - GPT-2

- Tokenization
  - BypePair Encoding
  - WordPiece Tokenizer

- Metrics
  - BLEU
  - ROUGE (-N, -L, -S)
  - Perplexity
  - METEOR
  - CIDER

- Datasets
  - IMDB Reviews Dataset
  - Flickr Dataset
  - NLTK POS Datasets (treebank, brown, conll2000)
  - SQuAD QA Dataset
  - Genius Lyrics Dataset
  - LAMBADA Dataset
  - Wiki en dataset
  - English to Telugu Translation Dataset

- Tasks
  - Sentiment Classification
  - POS Tagging
  - Image Captioning
  - Machine Translation
  - Question Answering
  - Text Generation

### Implementation Details

<table>
<thead>
<tr>
<th>Algorithm</th>
<th>Task</th>
<th>Tokenization</th>
<th>Output</th>
<th>Dataset</th>
</tr>
</thead>
<tbody><tr>
<td><strong>BOW</strong></td>
<td>Text Representation</td>
<td>Preprocessed words</td>
<td><ul><li>Text Label, Vector npy files</li><li>Top K Vocab Frequency Histogram png</li><li>Vocab frequency csv</li><li>Wordcloud png</li></ul></td>
<td>IMDB Reviews</td>
</tr>
<tr>
<td><strong>Ngram</strong></td>
<td>Text Representation</td>
<td>Preprocessed Words</td>
<td><ul><li>Text Label, Vector npy files</li><li>Top K Vocab Frequency Histogram png</li><li>Top K ngrams Piechart ong</li><li>Vocab frequency csv</li><li>Wordcloud png</li></ul></td>
<td>IMDB Reviews</td>
</tr>
<tr>
<td><strong>TF-IDF</strong></td>
<td>Text Representation</td>
<td>Preprocessed words</td>
<td><ul><li>Text Label, Vector npy files</li><li>TF PCA Pairplot png</li><li>TF-IDF PCA Pairplot png</li><li>IDF csv</li></ul></td>
<td>IMDB Reviews</td>
</tr>
<tr>
<td><strong>HMM</strong></td>
<td>POS Tagging</td>
<td>Preprocessed words</td>
<td><ul><li>Data Analysis png (sent len, POS tags count)</li><li>Emission Matrix TSNE html</li><li>Emission matrix csv</li><li>Test Predictions conf matrix, clf report png</li><li>Transition Matrix csv, png</li></ul></td>
<td>NLTK Treebank</td>
</tr>
<tr>
<td><strong>Word2Vec</strong></td>
<td>Text Representation</td>
<td>Preprocessed words</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Word Embeddings TSNE html</li></ul></td>
<td>IMDB Reviews</td>
</tr>
<tr>
<td><strong>GloVe</strong></td>
<td>Text Representation</td>
<td>Preprocessed words</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Word Embeddings TSNE html</li><li>Top K Cooccurence Matrix png</li></ul></td>
<td>IMDB Reviews</td>
</tr>
<tr>
<td><strong>RNN</strong></td>
<td>Sentiment Classification</td>
<td>Preprocessed words</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Word Embeddings TSNE html</li><li>Confusion Matrix png</li><li>Training History png</li></ul></td>
<td>IMDB Reviews</td>
</tr>
<tr>
<td><strong>LSTM</strong></td>
<td>Image Captioning</td>
<td>Preprocessed words</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Word Embeddings TSNE html</li><li>Training History png</li></ul></td>
<td>Flickr 8k</td>
</tr>
<tr>
<td><strong>GRU</strong></td>
<td>POS Tagging</td>
<td>Preprocessed words</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Word Embeddings TSNE html</li><li>Confusion Matrix png</li><li>Test predictions csv</li><li>Training History png</li></ul></td>
<td>NLTK Treebank, Broown, Conll2000</td>
</tr>
<tr>
<td><strong>Seq2Seq + Attention</strong></td>
<td>Machine Translation</td>
<td>Tokenization</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Source, Target Word Embeddings TSNE html</li><li>Test predictions csv</li><li>Training History png</li></ul></td>
<td>English to Telugu Translation</td>
</tr>
<tr>
<td><strong>Transformer</strong></td>
<td>Lyrics Generation</td>
<td>BytePairEncoding</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Token Embeddings TSNE html</li><li>Test predictions csv</li><li>Training History png</li></ul></td>
<td>Genius Lyrics</td>
</tr>
<tr>
<td><strong>BERT</strong></td>
<td>NSP Pretraining, QA Finetuning</td>
<td>WordPiece</td>
<td><ul><li>Best Model pt (pretrain, finetune)</li><li>Training History json (pretrain, finetune)</li><li>Token Embeddings TSNE html</li><li>Finetune Test predictions csv</li><li>Training History png (pretrain, finetune)</li></ul></td>
<td>Wiki en, SQuAD v1</td>
</tr>
<tr>
<td><strong>GPT-2</strong></td>
<td>Sentence Completition</td>
<td>BytePairEncoding</td>
<td><ul><li>Best Model pt</li><li>Training History json</li><li>Token Embeddings TSNE html</li><li>Test predictions csv</li><li>Training History png</li></ul></td>
<td>LAMBADA</td>
</tr>
</tbody></table>



## Examples 🌟

Run Train and Inference directly through import
```python
import yaml
from scratch_nlp.src.core.gpt import gpt

with open(config_path, "r") as stream:
  config_dict = yaml.safe_load(stream)

gpt = gpt.GPT(config_dict)
gpt.run()
```

Run through CLI
```bash
  cd src
  python main.py --config_path '<config_path>' --algo '<algo name>' --log_folder '<output folder>'
```

## Contributing 🤝

Contributions are always welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for ways to get started.

## Acknowledgements 💡

I have referred to so many online resources to create this project. I'm adding all the resources to [RESOURCES.md](RESOURCES.md). Thanks to all who has created those blogs/code/datasets 😊.

Thanks to [CS224N](https://web.stanford.edu/class/cs224n/) course which gave me motivation to start this project

## About Me 👤
I am Shanmukha Sainath, working as AI Engineer at KLA Corporation. I have done my Bachelors from Department of Electronics and Electrical Communication Engineering department with Minor in Computer Science Engineering and Micro in Artificial Intelligence and Applications from IIT Kharagpur. 

### Connect with me

<a href="https://linktr.ee/shanmukh05" target="blank"><img src="https://raw.githubusercontent.com/shanmukh05/scratch_nlp/main/assets/connect.png" alt="@shanmukh05" width="200"/></a>

## Lessons Learned 📌

Most of the things present in this project are pretty new to me. I'm listing down my major learnings when creating this project

- NLP Algorithms
- Research paper Implementation
- Designing Project structure
- Documentation 
- GitHub pages
- PIP packaging       

## License ⚖️

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Feedback 📣

If you have any feedback, please reach out to me at venkatashanmukhasainathg@gmail.com