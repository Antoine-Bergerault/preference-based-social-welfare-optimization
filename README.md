# Preference-based social welfare optimization

This repository contains the code for a semester research project done with the [Sycamore](https://www.epfl.ch/labs/sycamore/) lab at EPFL. The report for the project can be found [here](#).
We tackle the problem of the decentralized optimization of a global objective by combining mechanism design and preferential bayesian optimization.

## Configuring the environment

Run the following
```bash
conda env create -f environment.yml -n pref
conda activate pref

pip install -e .
```

## Running experiments

The configuration for running experiments is managed by [hydra](https://hydra.cc) in `configs`.

The entrypoint for all experiments is `app.py`.

For example, use the following command to run the main algorithm with a equilibrium oracle and an horizon of 10:
```
python app.py learning_algorithm=oracle horizon=10
```

For predefined experiments referenced in the report, use the following command:
```
python app.py -cd configs/experiments --config-name experiment-{number}
```

To perform more advanced inspections, you might want to create or use a script in `/scripts`.

## Running tests

We use [pytest](https://docs.pytest.org/en/8.0.x/) as our testing framework.

Run the tests using the following command:
```
cd tests && python -m pytest
```
