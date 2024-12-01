Documentation
=============

:doc:`Docs <modules>`

Installation
============

Install using pip
-----------------

.. code:: bash

      pip install ScratchNLP

Install Manually for development
--------------------------------

Clone the repo

.. code:: bash

     gh repo clone shanmukh05/scratch_nlp

Install dependencies

.. code:: bash

     pip install -r requirements.txt

Features
========

-  Algorithms

   -  Bag of Words
   -  Ngram
   -  TF-IDF
   -  Hidden Markov Model
   -  Word2Vec
   -  GloVe
   -  RNN (Many to One)
   -  LSTM (One to Many)
   -  GRU (Many to Many Synced)
   -  Seq2Seq + Attention (Many to Many)
   -  Transformer
   -  BERT
   -  GPT-2

-  Tokenization

   -  BypePair Encoding
   -  WordPiece Tokenizer

-  Metrics

   -  BLEU
   -  ROUGE (-N, -L, -S)
   -  Perplexity
   -  METEOR
   -  CIDER

-  Datasets

   -  IMDB Reviews Dataset
   -  Flickr Dataset
   -  NLTK POS Datasets (treebank, brown, conll2000)
   -  SQuAD QA Dataset
   -  Genius Lyrics Dataset
   -  LAMBADA Dataset
   -  Wiki en dataset
   -  English to Telugu Translation Dataset

-  Tasks

   -  Sentiment Classification
   -  POS Tagging
   -  Image Captioning
   -  Machine Translation
   -  Question Answering
   -  Text Generation

Implementation Details
----------------------

.. raw:: html

   <style>
   td {
       border: solid 2px lightgrey;
       text-align: center;
   }
   th {
       border: solid 2px lightgrey;
       text-align: center;
   }
   </style>

.. raw:: html

   <table>

.. raw:: html

   <thead>

.. raw:: html

   <tr>

.. raw:: html

   <th>

Algorithm

.. raw:: html

   </th>

.. raw:: html

   <th>

Task

.. raw:: html

   </th>

.. raw:: html

   <th>

Tokenization

.. raw:: html

   </th>

.. raw:: html

   <th>

Output

.. raw:: html

   </th>

.. raw:: html

   <th>

Dataset

.. raw:: html

   </th>

.. raw:: html

   </tr>

.. raw:: html

   </thead>

.. raw:: html

   <tbody>

.. raw:: html

   <tr>

.. raw:: html

   <td>

BOW

.. raw:: html

   </td>

.. raw:: html

   <td>

Text Representation

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Text Label, Vector npy files

.. raw:: html

   </li>

.. raw:: html

   <li>

Top K Vocab Frequency Histogram png

.. raw:: html

   </li>

.. raw:: html

   <li>

Vocab frequency csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Wordcloud png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

IMDB Reviews

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

Ngram

.. raw:: html

   </td>

.. raw:: html

   <td>

Text Representation

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed Words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Text Label, Vector npy files

.. raw:: html

   </li>

.. raw:: html

   <li>

Top K Vocab Frequency Histogram png

.. raw:: html

   </li>

.. raw:: html

   <li>

Top K ngrams Piechart ong

.. raw:: html

   </li>

.. raw:: html

   <li>

Vocab frequency csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Wordcloud png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

IMDB Reviews

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

TF-IDF

.. raw:: html

   </td>

.. raw:: html

   <td>

Text Representation

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Text Label, Vector npy files

.. raw:: html

   </li>

.. raw:: html

   <li>

TF PCA Pairplot png

.. raw:: html

   </li>

.. raw:: html

   <li>

TF-IDF PCA Pairplot png

.. raw:: html

   </li>

.. raw:: html

   <li>

IDF csv

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

IMDB Reviews

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

HMM

.. raw:: html

   </td>

.. raw:: html

   <td>

POS Tagging

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Data Analysis png (sent len, POS tags count)

.. raw:: html

   </li>

.. raw:: html

   <li>

Emission Matrix TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Emission matrix csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Test Predictions conf matrix, clf report png

.. raw:: html

   </li>

.. raw:: html

   <li>

Transition Matrix csv, png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

NLTK Treebank

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

Word2Vec

.. raw:: html

   </td>

.. raw:: html

   <td>

Text Representation

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Word Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

IMDB Reviews

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

GloVe

.. raw:: html

   </td>

.. raw:: html

   <td>

Text Representation

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Word Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Top K Cooccurence Matrix png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

IMDB Reviews

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

RNN

.. raw:: html

   </td>

.. raw:: html

   <td>

Sentiment Classification

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Word Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Confusion Matrix png

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

IMDB Reviews

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

LSTM

.. raw:: html

   </td>

.. raw:: html

   <td>

Image Captioning

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Word Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

Flickr 8k

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

GRU

.. raw:: html

   </td>

.. raw:: html

   <td>

POS Tagging

.. raw:: html

   </td>

.. raw:: html

   <td>

Preprocessed words

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Word Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Confusion Matrix png

.. raw:: html

   </li>

.. raw:: html

   <li>

Test predictions csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

NLTK Treebank, Broown, Conll2000

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

Seq2Seq + Attention

.. raw:: html

   </td>

.. raw:: html

   <td>

Machine Translation

.. raw:: html

   </td>

.. raw:: html

   <td>

Tokenization

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Source, Target Word Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Test predictions csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

English to Telugu Translation

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

Transformer

.. raw:: html

   </td>

.. raw:: html

   <td>

Lyrics Generation

.. raw:: html

   </td>

.. raw:: html

   <td>

BytePairEncoding

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Token Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Test predictions csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

Genius Lyrics

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

BERT

.. raw:: html

   </td>

.. raw:: html

   <td>

NSP Pretraining, QA Finetuning

.. raw:: html

   </td>

.. raw:: html

   <td>

WordPiece

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt (pretrain, finetune)

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json (pretrain, finetune)

.. raw:: html

   </li>

.. raw:: html

   <li>

Token Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Finetune Test predictions csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png (pretrain, finetune)

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

Wiki en, SQuAD v1

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

GPT-2

.. raw:: html

   </td>

.. raw:: html

   <td>

Sentence Completition

.. raw:: html

   </td>

.. raw:: html

   <td>

BytePairEncoding

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   <ul>

.. raw:: html

   <li>

Best Model pt

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History json

.. raw:: html

   </li>

.. raw:: html

   <li>

Token Embeddings TSNE html

.. raw:: html

   </li>

.. raw:: html

   <li>

Test predictions csv

.. raw:: html

   </li>

.. raw:: html

   <li>

Training History png

.. raw:: html

   </li>

.. raw:: html

   </ul>

.. raw:: html

   </td>

.. raw:: html

   <td>

LAMBADA

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </tbody>

.. raw:: html

   </table>

Examples
========

Run Train and Inference directly through import

.. code:: python

   import yaml
   from scratch_nlp.src.core.gpt import gpt

   with open(config_path, "r") as stream:
     config_dict = yaml.safe_load(stream)

   gpt = gpt.GPT(config_dict)
   gpt.run()

Run through CLI

.. code:: bash

     cd src
     python main.py --config_path '<config_path>' --algo '<algo name>' --log_folder '<output folder>'

Contributing
============

Contributions are always welcome!

See `CONTRIBUTING.md <CONTRIBUTING.md>`__ for ways to get started.

Acknowledgements
================

I have referred to sa many online resources to create this project. I’m
adding all the resources to `RESOURCES.md <RESOURCES.md>`__. Thanks to
all who has created those blogs/code/datasets.

Thanks to `CS224N <https://web.stanford.edu/class/cs224n/>`__ course
which gave me motivation to start this project

About Me
========

I am Shanmukha Sainath, working as AI Engineer at KLA Corporation. I
have done my Bachelors from Department of Electronics and Electrical
Communication Engineering department with Minor in Computer Science
Engineering and Micro in Artificial Intelligence and Applications from
IIT Kharagpur.

Connect with me
---------------

.. figure:: https://raw.githubusercontent.com/shanmukh05/scratch_nlp/main/assets/connect.png
   :alt: Logo
   :width: 200px
   :target: https://linktr.ee/shanmukh05


Lessons Learned
===============

Most of the things present in this project are pretty new to me. I’m
listing down my major learnings when creating this project

-  NLP Algorithms
-  Research paper Implementation
-  Designing Project structure
-  Documentation
-  GitHub pages
-  PIP packaging

License
=======

|MIT License|

Feedback
========

If you have any feedback, please reach out to me at
venkatashanmukhasainathg@gmail.com

.. |MIT License| image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://choosealicense.com/licenses/mit/
