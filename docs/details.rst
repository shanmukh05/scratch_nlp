Documentation
=============

:doc:`Docs <modules>`

Installation
============

Install using pip
-----------------

.. code:: bash

      pip install scratch-nlp

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

========= ==== ============ ====== =======
Algorithm Task Tokenization Output Dataset
========= ==== ============ ====== =======
========= ==== ============ ====== =======

\| **BOW** \| Text Representation \| Preprocessed words \|

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

\| IMDB Reviews \| Data \| \| **Ngram** \| Text Representation \|
Preprocessed Words \|

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

::

    | IMDB Reviews     | Data     |

\| **TF-IDF** \| Text Representation \| Preprocessed words \|

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

\| IMDB Reviews \| \| **HMM** \| Text Representation \| Preprocessed
words \|

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

\| IMDB Reviews \| \| **Word2Vec** \| Text Representation \|
Preprocessed words \|

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

\| IMDB Reviews \| \| **GloVe** \| Text Representation \| Preprocessed
words \|

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

\| IMDB Reviews \| \| **RNN** \| Sentiment Classification \|
Preprocessed words \|

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

\| IMDB Reviews \| \| **LSTM** \| Image Captioning \| Preprocessed words
\|

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

\| Flickr 8k \| \| **GRU** \| POS Tagging \| Preprocessed words \|

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

\| NLTK Treebank, Broown, Conll2000 \| \| **Seq2Seq + Attention** \|
Machine Translation \| Tokenization \|

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

\| English to Telugu Translation \| \| **Transformer** \| Lyrics
Generation \| BytePairEncoding \|

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

\| Genius Lyrics \| \| **BERT** \| NSP Pretraining, QA Finetuning \|
WordPiece \|

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

\| Wiki en, SQuAD v1 \| \| **GPT-2** \| Sentence Completition \|
BytePairEncoding \|

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

\| LAMBADA \|

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
