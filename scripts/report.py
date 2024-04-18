import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from pref import pref_social_welfare

def run_and_store_results(cfg: DictConfig, output_dir):
    csv_output = output_dir / "results.csv"
    
    t = 0
    ucb_rewards_hist = []
    rewards_hist = []
    preferences = 0
    
    for (s_ucb, actions, ucb_rewards, rewards, preference) in pref_social_welfare(cfg, generator=True):
        t += 1
        ucb_rewards_hist.append(ucb_rewards)
        rewards_hist.append(rewards)
        preferences += preference
    
    results = pd.Series({
        "time_to_convergence": t,
        "preference_ratio": preferences / t,
        "rewards_hist": rewards_hist,
        "ucb_rewards_hist": rewards_hist
    })
    
    results.to_csv(csv_output)
    
    return results

def load_and_compare_results(hydra_config, cfg: DictConfig, output_dir, results: pd.Series, comparisons_dir):
    keys = list(hydra_config.sweeper["params"].keys())
    
    # add keys where no cross-product over parameters is needed
    if "+jobs_config" in keys:
        other_keys = list(cfg["jobs_config"].keys())
        keys.remove("+jobs_config")
        keys += other_keys
        
    run_fingerprint = {key: cfg[key] for key in keys}
    
    report_graph = cfg.get("report_add_graphs", True)
    
    # few files describing the results of the entire experiment
    registry_file = comparisons_dir / "registry.csv"
    all_results_file = comparisons_dir / "all_results.pkl"
    latex_table_file = comparisons_dir / "latex_table.tex"
    graphs_dir = comparisons_dir / "graph"
    
    lower_is_better_cols = ["time_to_convergence"]
    higher_is_better_cols = ["preference_ratio"]
    
    graph_cols = ["rewards_hist", "ucb_rewards_hist"]
    
    if not os.path.isfile(registry_file):
        with open(registry_file, 'w', newline='') as file:
            writer = csv.writer(file)
            fields = keys + ["output_dir"]
            
            writer.writerow(fields)
        
        if report_graph:
            Path(graphs_dir).mkdir(parents=True, exist_ok=False)
           
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
            index=report_graph, # add index so we can refer to indices in the graphs 
            na_rep="-",
            float_format="%.2f"
        ))

    if report_graph:
        for col in graph_cols:
            graph_groups = all_results[col]
            
            graph_data = None
            for index, run_data in graph_groups.items():
                horizon = len(run_data)
                # the x-axis always represents the turns/time
                x = np.arange(horizon)
                
                group_df = pd.DataFrame({
                    "index": [index]*horizon,
                    "turn": x,
                    col: run_data
                })
                
                if graph_data is None:
                    graph_data = group_df
                else:
                    graph_data = pd.concat([graph_data, group_df], ignore_index=True)
            
            fig = px.line(graph_data, x="turn", y=col, color="index", title=f"{col} through time")
            fig.write_image(graphs_dir / f"{col}.png")

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