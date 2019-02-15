# Brexit classifier
An artifical neural network for classifying text comments as being either pro or anti Brexit.

## Setup
Dependencies are listed in the `Pipfile`, and given that you have `pipenv` setup, all you need to do is to run `pipenv install` to install them all. You can then run all the commands given below inside a virtual environment, using `pipenv shell` (it is also possible to prefix all commands below with `pipenv run `).

**NB** You _might_ have to swap `tensorflow-gpu` to just `tensorflow` within the `Pipfile` if your computer does not have a GPU, I'm not sure about this however.

## Usage:
`$ python main.py <command> <arg>`

Available commands:
    
- `test <path to test set>`
- `predict "Some sentence to predict"`
- `explore`

### `test`
Example:
```
$ python main.py test "some/path/to/test-data.tsv"
```
Evaluates the model against the annotated data given in the path.

### `predict`
Example:
```
$ python main.py predict "Brexiters suck"
```
Predict the classification of a given sentence about Brexit

### `explore`
**NB** This requires the GloVe embeddings matrices to be downloaded, run:
```
$ chmod +x download_glove.sh
$ ./download_glove.sh
```
This might take a while, these matrices are quite large.

Then you can run:
```
$ python main.py explore
```
The best recorded accuracy will be saved in `exploration_best.txt`, and all explored configurations will be saved in `resploration_results.json` together with their accuracy. The best model will also be saved into the `models` directory so that it can be re-used later.
