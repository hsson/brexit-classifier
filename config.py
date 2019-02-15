import random

def random_config():
    return {
        'tokenizer_size': random.randrange(5000, 175000),
        'tokenizer_lowercase': random.random() > 0.5,
        'max_length': random.randrange(15, 110),
        'use_glove': random.random() > 0.5,
        'glove_source': random.sample(['twitter', 'wiki'], 1)[0],
        'glove_size': random.sample([50, 100, 200], 1)[0],
        'glove_trainable': random.random() > 0.5,
        'embedding_output': random.randrange(20, 300),
        'lstm_cells': random.randrange(10, 80),
        'lstm_dropout': random.uniform(0, 0.2),
        'dropout_1': random.uniform(0, 0.5),
        'dense_units': random.randrange(25, 750),
        'dropout_2': random.uniform(0, 0.5),
        'dense_use_2': random.random() > 0.5,
        'dense_units_2': random.randrange(25, 750),
        'dropout_3': random.uniform(0, 0.5),
        'dense_use_3': random.random() > 0.5,
        'dense_units_3': random.randrange(25, 750),
        'dropout_4': random.uniform(0, 0.5),
        'optimizer_lr': random.uniform(0.0001, 0.001),
        'batch_size': random.randrange(20, 100)
    }

def config_generator():
    while True:
        yield random_config()
