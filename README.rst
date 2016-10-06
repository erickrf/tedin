Recognizing Textual Entailment
==============================

This project contains some algorithms for performing the RTE task in Portuguese.

Configuration
-------------

The file ``config.py`` contains variables with paths to some specific data
files (e.g., wordnet, `DELAF dictionary`_) and endpoints (stanford corenlp).

.. _`DELAF dictionary`: http://www.nilc.icmc.usp.br/nilc/projects/unitex-pb/web/dicionarios.html

Scripts that can be run:

* ``preprocess.py``: tokenize the input and run a dependency parser on it

* ``pilotrte.py``: train a model with previously processed data.

TODO: improve the documentation
