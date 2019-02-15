import json
import sys
import pickle
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.initializers import RandomNormal
from keras.layers import Dense, Dropout, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from config import config_generator

# fix random seed for reproducibility
seed = 812345
np.random.seed(seed)

data_path = 'data/brexit-ann.tsv'
results_file = 'exploration_results.json'
results_best_file = 'exploration_best.txt'
test_size = 0.2
glove = {
    'twitter': {
        50: 'glove.twitter.27B.50d.txt',
        100: 'glove.twitter.27B.100d.txt',
        200: 'glove.twitter.27B.200d.txt'
    },
    'wiki': {
        50: 'glove.6B.50d.txt',
        100: 'glove.6B.100d.txt',
        200: 'glove.6B.200d.txt'
    }
}

def read_tsv_annotated(path):
    dataset = pd.read_csv(path, sep='\t', header=None)
    dataset = dataset.values
    np.random.shuffle(dataset)

    def filter_row(row):
        if '-1' in str(row[0]):
            return False
        labels = np.array(str(row[0]).split('/'), dtype='float32')
        count = len(labels)
        label_sum = np.sum(labels, dtype='float32')
        return not abs(label_sum - (count/2.0)) < 0.001

    dataset = dataset[np.array([filter_row(row) for row in dataset])]
    for row in dataset:
        labels = np.array(str(row[0]).split('/'), dtype='float32')
        label_sum = np.sum(labels)
        if label_sum > len(labels)/2.0:
            row[0] = 1
        else:
            row[0] = 0
    return dataset[:, 1], dataset[:, 0]

def save_model(model, tokenizer, config,  name='model'):
    model_json = model.to_json()
    with open ("models/{0}.json".format(name), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("models/{0}.h5".format(name))
    print("Saved model to files")

    with open('models/tokenizer-{0}.pickle'.format(name), 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved tokenizer to file")

    with open('models/config-{0}.json'.format(name), 'w') as config_file:
        json.dump(config, config_file)
    print('Saved model config to file')

def load_model(name='model'):
    model_json = None
    with open("models/{0}.json".format(name), "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("models/{0}.h5".format(name))
    print("Loaded model")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    with open('models/tokenizer-{0}.pickle'.format(name), 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    print("Loaded tokenizer")

    with open('models/config-{0}.json'.format(name)) as config_file:
        config = json.load(config_file)
    print('Loaded config')

    return model, tokenizer, config

def build_model(config, vocab_size, tokenizer):
    model = Sequential()
    if config['use_glove']:
        source = config['glove_source']
        glove_size = config['glove_size']
        glove_path = 'embeddings/{0}'.format(glove[source][glove_size])
        embedding_index = load_word_embeddings(glove_path)
        embedding_matrix = create_embedding_matrix(embedding_index, vocab_size, config['glove_size'], tokenizer)
        model.add(Embedding(vocab_size, config['glove_size'], weights=[embedding_matrix], input_length=config['max_length'], trainable=config['glove_trainable']))
    else:
        model.add(Embedding(vocab_size, config['embedding_output'], mask_zero=True))
    model.add(LSTM(config['lstm_cells'], recurrent_dropout=config['lstm_dropout'], recurrent_activation='sigmoid'))
    model.add(Dropout(config['dropout_1']))
    model.add(Dense(units=config['dense_units'], activation='sigmoid', kernel_initializer=RandomNormal(seed=seed)))
    if config['dense_use_2']:
        model.add(Dropout(config['dropout_2']))
        model.add(Dense(units=config['dense_units_2'], activation='sigmoid', kernel_initializer=RandomNormal(seed=seed*2)))
    if config['dense_use_3']:
        model.add(Dropout(config['dropout_3']))
        model.add(Dense(units=config['dense_units_3'], activation='sigmoid', kernel_initializer=RandomNormal(seed=seed/2)))
    model.add(Dropout(config['dropout_4']))
    model.add(Dense(units=1, activation='sigmoid'))

    opt = Adam(lr=config['optimizer_lr'])

    model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
    return model

def load_word_embeddings(path):
    embeddings_index = dict()
    with open(path) as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index

def create_embedding_matrix(embeddings_index, vocab_size, embedding_size, tokenizer):
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def predict_sentence(sentence, model_name='best'):
    X = np.array([sentence])
    model, tokenizer, config = load_model(name=model_name)
    X_proc = process_input(X, tokenizer, config)
    prediction = model.predict_classes(X_proc)
    print("Predicted: {0}".format(prediction[0][0]))

def test_model(test_set_path, model_name='best'):
    model, tokenizer, config = load_model(name=model_name)
    X, Y = read_tsv_annotated(test_set_path)
    X = process_input(X, tokenizer, config)
    _, acc = model.evaluate(X, Y)
    print("Accuracy: {0}".format(acc))

def process_input(X, tokenizer, config):
    vocab_size = len(tokenizer.word_index) + 1
    encoded = tokenizer.texts_to_sequences(X)
    max_length = config['max_length']
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def train_model(config, X_train, X_test, Y_train, Y_test, verbose=0):
    t = Tokenizer(num_words=config['tokenizer_size'], lower=config['tokenizer_lowercase'])
    t.fit_on_texts(X_train)

    train = process_input(X_train, t, config)
    test = process_input(X_test, t, config)

    vocab_size = len(t.word_index) + 1
    model = build_model(config, vocab_size, t)

    es = EarlyStopping(
            monitor='val_acc',
            min_delta=0,
            patience=2,
            verbose=0,
            mode='auto',
            restore_best_weights=True)

    model.fit(
            train,
            Y_train,
            verbose=verbose,
            epochs=200,
            batch_size=config['batch_size'],
            validation_data=(test, Y_test),
            callbacks=[es])

    return model, t

def explore_models():
    X, Y = read_tsv_annotated(data_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    models_trained = 0
    for config in config_generator():
        print('\nTraining model #{0}'.format(models_trained + 1))

        model, tokenizer = train_model(config, X_train, X_test, Y_train, Y_test)
        data_test = process_input(X_test, tokenizer, config)
        _, accuracy = model.evaluate(data_test, Y_test)
        models_trained += 1
        print('Accuracy: %f %%' % (accuracy*100))
        result = {
            'accuracy': accuracy,
            'params': config
        }
        with open(results_file, 'a') as f:
            f.write(json.dumps(result) + '\n')

        got_best = False
        with open(results_best_file, 'r+') as f:
            current_best = float(f.read())
            if accuracy > current_best:
                got_best = True
                f.seek(0)
                f.write(str(accuracy))
        if got_best:
            save_model(model, tokenizer, config, name='exploration')

_usage = """
Usage:
    test <path to test set>
    predict <sentence to predict>
    explore # This will explore/train new models
"""
def main(args):
    if len(args) <= 1:
        print(_usage)
        return
    command = args[1]
    if command == 'predict':
        if len(args) != 3:
            print('Missing required sentence argument')
            print('Example: predict "The EU is worthless"')
            sys.exit(1)
            return
        arg = args[2]
        predict_sentence(arg)
    elif command == 'test':
        if len(args) != 3:
            print('Missing required path argument')
            sys.exit(1)
            return
        arg = args[2]
        test_model(arg)
    elif command == 'explore':
        explore_models()
    else:
        print('Unknown command "{0}"'.format(command))
        sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)

