import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

import csv
import os
from pathlib import Path

import pandas as pd

from pref import pref_social_welfare

def run_and_store_results(cfg: DictConfig, output_dir):
    csv_output = output_dir / "results.csv"
    
    t = 0
    rewards_hist = []
    preferences = 0
    
    for (s_ucb, actions, ucb_rewards, rewards, preference) in pref_social_welfare(cfg, generator=True):
        t += 1
        rewards_hist.append(rewards)
        preferences += preference
    
    results = pd.Series({
        "time_to_convergence": t,
        "preference_ratio": preferences / t
    })
    
    results.to_csv(csv_output)
    
    return results

def load_and_compare_results(hydra_config, cfg: DictConfig, output_dir, results: pd.Series, comparisons_dir):
    keys = list(hydra_config.sweeper["params"].keys())
    run_fingerprint = {key: cfg[key] for key in keys}
    
    # few files describing the results of the entire experiment
    registry_file = comparisons_dir / "registry.csv"
    all_results_file = comparisons_dir / "all_results.pkl"
    latex_table_file = comparisons_dir / "latex_table.tex"
    
    lower_is_better_cols = ["time_to_convergence"]
    higher_is_better_cols = ["preference_ratio"]
    
    if not os.path.isfile(registry_file):
        with open(registry_file, 'w', newline='') as file:
            writer = csv.writer(file)
            fields = keys + ["output_dir"]
            
            writer.writerow(fields)
           
    with open(registry_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list(run_fingerprint.values()) + [output_dir])
    
    results = pd.Series(results.to_dict() | run_fingerprint) # append fingerprint to results
    results_df = results.to_frame().T
    if os.path.isfile(all_results_file):
        all_results = pd.read_pickle(all_results_file)
        all_results = pd.concat([all_results, results_df], ignore_index=True) # add new results to the comparison
    else:
        all_results = results_df

    all_results.to_pickle(all_results_file)
    
    comparisons = all_results[
        keys +
        lower_is_better_cols +
        higher_is_better_cols
    ]
    
    with open(latex_table_file, "w", newline='') as file:
        file.write(comparisons.to_latex(
            index=False, 
            na_rep="-",
            float_format="%.2f"
        ))

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def my_app(cfg: DictConfig) -> None:
    hydra_config = HydraConfig.get()
    run_mode = hydra_config.mode
    output_dir = hydra_config.runtime.output_dir
    
    print("Experiment configuration loaded.")
    print("Results will be stored in:", output_dir)
    
    results = run_and_store_results(cfg, output_dir)
    
    if run_mode == RunMode.MULTIRUN:
        parent_dir = Path(output_dir).parent.resolve()
        comparisons_dir = parent_dir / "results"
        
        # create directory if it does not already exist
        Path(comparisons_dir).mkdir(parents=True, exist_ok=True)
        
        load_and_compare_results(hydra_config, cfg, output_dir, results, comparisons_dir)
        
        print("Comparisons for multirun saved in:", comparisons_dir)
        print()

if __name__ == "__main__":
    my_app()