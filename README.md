# OptML_ASGD

This project aims at studying the performance of ASGD algorithms and variants with different settings.

## Requirements

    pip install -r requirements.txt

## Scripts

`src/sgd.py`: implement a classic sequential SGD training algorithm = reference/baseline

To run a training with default parameters, run the following command:

    python -m src.sgd

Some options can be passed to the script and are explained when typing:

    python -m src.sgd --help

`src/asgd.py`: emulates an ASGD training algorithm

To run a training with default parameters, run the following command:

    python -m src.asgd

Some options can be passed to the script and are explained when typing:

    python -m src.asgd --help

`src/data.py`: implement some data loading utilities

`src/models.py`: implement some models and associated datasets

## Results

Tensorboard logs containing the results of the training presented in the report can be found in the `runs` directory.

These logs can be visualized using the following command (when executed at the root of the project):

    tensorboard --logdir runs

Since there are a lot of runs, tensorboard may be a bit slow to parse them. One may want to first select a subset of these runs to visualize. This can be done manually by copying the logs to a new directory.

### Reproducibility

To reproduce exactly our results, one can find the slurm scripts we ran in the `scripts` directory.
