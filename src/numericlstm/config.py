# -*- coding: utf-8 -*-

"""
Variables for parametrizing and paths to persist models.
"""

num_time_steps = 9
embedding_size = 300

# get the data
train_size = 32000
valid_size = 1000

batch_size = 32
num_epochs = 1

accumulated_loss = 0
report_interval = 100
save_path = '../checkpoints/basic-memorizer.dat'

