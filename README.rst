Infernal
========

Infernal (INFERence in NAtural Language) is a model for performing natural
language inference / recognizing textual entailment based on handcrafted
features. It was implemented primarily for Portuguese, but most of it can be
reused for other languages.

Reference
---------

If you publish research using or expanding on Infernal, please cite:

Erick Fonseca and Sandra M. Aluísio. Syntactic Knowledge for Natural Language
Inference in Portuguese. In: Proceedings of the 2018 International Conference
on the Computational Processing of Portuguese (PROPOR). 2018.
*(accepted for publication)*

```
@inproceedings{infernal,
  author = {Erick Fonseca and Sandra M. Alu\'isio},
  title = {{Syntactic Knowledge for Natural Language Inference in Portuguese}},
  year = {2018},
  booktitle = {Proceedings of the 2018 International Conference
on the Computational Processing of Portuguese (PROPOR)}
}
```

Configuration
-------------

The file ``config.py`` contains variables with paths to some specific data
files (e.g., wordnet, `DELAF dictionary`_) and endpoints (stanford corenlp).

.. _`DELAF dictionary`: http://www.nilc.icmc.usp.br/nilc/projects/unitex-pb/web/dicionarios.html

Scripts that can be run:

* ``preprocess.py``: tokenize the input and run a dependency parser on it

* ``pilotrte.py``: train a model with previously processed data.

TODO: improve the documentation
