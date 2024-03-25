# Sycamore project

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

For example, use the following command to run the main algorithm with a equilibrium oracle and an horizon of 10
```
python app.py learning_algorithm=oracle horizon=10
```