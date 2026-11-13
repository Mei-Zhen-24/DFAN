import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_benchmark import *

seeds = [2021]
for i, seed in enumerate(seeds):
    sys.argv = [
        "run_benchmark.py",
        "--config-path",
        "rolling_forecast_config.json",
        "--data-name-list",
        "ETTh1.csv",
        "--strategy-args",
        '{"horizon":96}',
        "--model-name",
        "dfan.DFAN",
        "--model-hyper-params",
        '{"horizon": 96, "loss": "MAE", "lradj": "type1", "norm": true, "num_epochs": 1, "patience": 5, "d_model": 512, "fc_dropout": 0.1, "n_heads": 1, "seq_len": 512, "batch_size": 32, "dropout": 0.5, "patch_len": 256, "rule_size": 2, "e_layers": 1, "num_experts": 2, "k": 1, "lr": 0.001}',
        "--gpus",
        "0",
        "--num-workers",
        "0",
        "--timeout",
        "60000",
        "--save-path",
        "ETTh1/DFAN",
        "--seed",
        f"{seed}",
    ]
    run_benchmark()
