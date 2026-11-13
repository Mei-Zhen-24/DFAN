# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, NoReturn
from datetime import datetime
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from ts_benchmark.utils.get_file_name import get_unique_file_suffix
from ts_benchmark.report import report
from ts_benchmark.common.constant import CONFIG_PATH, THIRD_PARTY_PATH
from ts_benchmark.pipeline import pipeline
from ts_benchmark.utils.parallel import ParallelBackend

sys.path.insert(0, THIRD_PARTY_PATH)
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd
from datetime import datetime


def str_to_bool(value: str) -> bool:
    """
    Converts a string to a boolean: True for 'True', '1', or 'T'; False for 'False', '0', or 'F'.
    """
    if value.lower() in ["true", "1", "t"]:
        return True
    elif value.lower() in ["false", "0", "f"]:
        return False
    else:
        raise ValueError("Invalid boolean value. Please enter 'True' or 'False'.")


def build_data_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the data loader config from commandline arguments and configuration dict
    """
    data_config = config_data["data_config"]
    data_config["data_name_list"] = args.data_name_list
    if args.data_set_name is not None:
        data_config["data_set_name"] = args.data_set_name
    return data_config


def build_model_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the model config from commandline arguments and configuration dict
    """
    model_config = config_data.get("model_config", None)

    if args.adapter is not None:
        args.adapter = [None if item == "None" else item for item in args.adapter]
        if len(args.model_name) > len(args.adapter):
            args.adapter.extend([None] * (len(args.model_name) - len(args.adapter)))
    else:
        args.adapter = [None] * len(args.model_name)

    if args.model_hyper_params is not None:
        args.model_hyper_params = [None if item == "None" else item for item in args.model_hyper_params]
        if len(args.model_name) > len(args.model_hyper_params):
            args.model_hyper_params.extend([None] * (len(args.model_name) - len(args.model_hyper_params)))
    else:
        args.model_hyper_params = [None] * len(args.model_name)

    for adapter, model_name, model_hyper_params in zip(args.adapter, args.model_name, args.model_hyper_params):
        model_config["models"].append(
            {
                "adapter": adapter,
                "model_name": model_name,
                "model_hyper_params": (json.loads(model_hyper_params) if model_hyper_params is not None else {}),
            }
        )

    return model_config


def build_evaluation_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the evaluation config from commandline arguments and configuration dict
    """
    evaluation_config = config_data["evaluation_config"]
    evaluation_config["save_path"] = args.save_path

    metric_list = []
    if args.metrics != "all" and args.metrics is not None:
        for metric in args.metrics:
            metric = json.loads(metric)
            metric_list.append(metric)
        evaluation_config["metrics"] = metric_list

    default_strategy_args = evaluation_config["strategy_args"]
    strategy_args_updates = json.loads(args.strategy_args) if args.strategy_args else None

    if strategy_args_updates is not None:
        default_strategy_args.update(strategy_args_updates)

    if args.seed is not None:
        default_strategy_args["seed"] = args.seed
    if args.save_true_pred is not None:
        default_strategy_args["save_true_pred"] = args.save_true_pred
    default_strategy_args["deterministic"] = args.deterministic

    return evaluation_config


def build_report_config(args: argparse.Namespace, config_data: Dict) -> Dict:
    """
    Builds the report config from commandline arguments and configuration dict
    """
    report_config = config_data["report_config"]
    report_config["aggregate_type"] = args.aggregate_type
    report_config["save_path"] = args.save_path

    return report_config


def init_worker(env: Dict) -> NoReturn:
    """
    An initializer function for each worker that does some global setup
    """
    sys.path.insert(0, THIRD_PARTY_PATH)
    torch.set_num_threads(1)


def run_benchmark():
    parser = argparse.ArgumentParser(
        description="run_benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # script name
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Evaluation config file path",
    )

    parser.add_argument(
        "--data-name-list",
        type=str,
        nargs="+",
        default=None,
        help="List of series names entered by the user",
    )

    parser.add_argument(
        "--data-set-name",
        type=str,
        nargs="+",
        default=None,
        help="List of dataset name names entered by the user," "only takes effect when data_name_list is not specified",
    )

    # model_config
    parser.add_argument(
        "--adapter",
        type=str,
        nargs="+",
        default=None,
        help="Adapter used to adapt the method to our pipeline",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        nargs="+",
        required=True,
        help="The relative path of the model that needs to be evaluated",
    )
    parser.add_argument(
        "--model-hyper-params",
        type=str,
        nargs="+",
        default=None,
        help=(
            "The input parameters corresponding to the models to be evaluated "
            "should correspond one-to-one with the --model-name options."
        ),
    )

    # evaluation_config
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Evaluation metrics that need to be calculated",
    )
    parser.add_argument(
        "--strategy-args",
        type=str,
        default=None,
        help="Parameters required for evaluating strategies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed that is set before evaluating any model-series pair, "
        "by default, use the seed value in the config file",
    )
    parser.add_argument(
        "--deterministic",
        type=str,
        default="efficient",
        choices=["full", "efficient", "none"],
        help="Specify the type of deterministic behavior for the algorithm. Options are: "
        "'full': Enables full deterministic mode. "
        "'efficient': Fixes only some seeds for efficiency. "
        "'none': No deterministic behavior is applied.",
    )

    # evaluation engine
    parser.add_argument(
        "--eval-backend",
        type=str,
        default="sequential",
        choices=["sequential", "ray"],
        help="Evaluation backend, use ray for parallel evaluation",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="Number of cpus to use, only available in both backends",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        help="List of gpu devices to use, only available in ray backends",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
        help="Number of evaluation workers",
    )
    # TODO: should timeout be part of the configuration file?
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Time limit for each evaluation task, in seconds",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=100,
        help="Max tasks to run on a single worker when using parallel backends",
    )
    # report_config
    parser.add_argument(
        "--aggregate_type",
        default="mean",
        help="Select the baseline algorithm to compare",
    )
    parser.add_argument(
        "--report-method",
        type=str,
        default="csv",
        choices=[
            "dash",
            "csv",
        ],
        help="Presentation form of algorithm performance comparison results",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="The relative path for saving evaluation results, relative to the result folder",
    )
    parser.add_argument(
        "--save-true-pred",
        type=str_to_bool,
        default=None,
        help="If true, saves the model's prediction results " "and the true values in evaluation result file",
    )
    args = parser.parse_args()
    args.save_path = os.path.join("csv", args.save_path)
    # 创建 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ## add
    args_dict = vars(args)  # 转为字典
    # 保存路径
    save_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, "args.json")
    # 保存为 JSON
    with open(save_file, "w") as f:
        json.dump(args_dict, f, indent=4)
    with open(save_file) as f:
        loaded_args = json.load(f)

    # 模型名
    model_full = loaded_args["model_name"][0]
    model_name = model_full.rsplit(".", 1)[-1]
    # 数据集名
    data_full = loaded_args["data_name_list"][0]
    data_name = data_full.rsplit(".", 1)[0]

    # 超参数
    hyper_params_str = loaded_args["model_hyper_params"][0]
    hyper_params = json.loads(hyper_params_str)
    seq_len = hyper_params["seq_len"]
    pred_len = hyper_params["horizon"]

    # 日志文件名
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    log_filename = f"{model_name}-{data_name}-{seq_len}-{pred_len}-{timestamp}.log"
    # 把日志存放在 logs/{model_name}/ 目录下
    log_dir = os.path.join("logs", model_name, data_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # 文件 handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s(%(lineno)d): %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 添加 handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    ## add end
    torch.set_num_threads(3)
    with open(os.path.join(CONFIG_PATH, args.config_path), "r") as file:
        config_data = json.load(file)

    required_configs = [
        "data_config",
        "model_config",
        "evaluation_config",
        "report_config",
    ]
    for config_name in required_configs:
        if config_data.get(config_name) is None:
            raise ValueError(f"{config_name} is none")

    data_config = build_data_config(args, config_data)
    model_config = build_model_config(args, config_data)
    evaluation_config = build_evaluation_config(args, config_data)
    report_config = build_report_config(args, config_data)

    ParallelBackend().init(
        backend=args.eval_backend,
        n_workers=args.num_workers,
        n_cpus=args.num_cpus,
        gpu_devices=args.gpus,
        default_timeout=args.timeout,
        max_tasks_per_child=args.max_tasks_per_child,
        worker_initializers=[init_worker],
    )
    try:
        pipeline(
            data_config,
            model_config,
            evaluation_config,
        )
    finally:
        ParallelBackend().close(force=True)
    # report_config["log_files_list"] = log_filenames
    # if args.report_method == "csv":
    #     filename = get_unique_file_suffix()
    #     leaderboard_file_name = "test_report" + filename
    #     report_config["leaderboard_file_name"] = leaderboard_file_name
    # report(report_config, report_method=args.report_method)


if __name__ == "__main__":
    run_benchmark()
