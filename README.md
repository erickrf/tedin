TEDIN
=====

TEDIN (Tree Edit DIstance Network) is a neural model for learning dynamic
tree edit (TED) operation costs. It is implemented in a way to solve Natural
Language Inference problems via TED.

This code includes a previous version of [Infernal](https://github.com/erickrf/infernal),
but in order to use that system, it is advisable to refer to its dedicated repository.


Configuration
-------------

The file ``config.py`` contains variables with paths to some specific data
files (e.g., wordnet, [DELAF dictionary](http://www.nilc.icmc.usp.br/nilc/projects/unitex-pb/web/dicionarios.html)) 
and endpoints (stanford corenlp).

The code is somewhat hard coded to work with Portuguese data. 


Usage
-----

Before running any of the scripts, set the environment variable `PYTHONPATH` to include the current directory:

```
export PYTHONPATH=.
```

In order to use TEDIN, perform the following steps:

1. Run the `preprocess.py` scripts with XML files containing the data. It will
    need a running installation of the CoreNLP parser.
1. Run `train-ranker.py` with the trained data. This trains the model with
    distant supervision to learn operation costs.
1. Run `train-classifier.py` to train an actual NLI classifier from the model trained
    in the previous step.
