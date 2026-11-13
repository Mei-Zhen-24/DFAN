# -*- coding: utf-8 -*-
from dataclasses import dataclass
from functools import reduce
from operator import and_
from typing import List, Dict, Type, Optional
import pandas as pd
import logging
from ts_benchmark.data.data_source import (
    LocalForecastingDataSource,
    DataSource,
)
from ts_benchmark.data.suites.global_storage import GlobalStorageDataServer
from ts_benchmark.evaluation.evaluate_model import eval_model
from ts_benchmark.models import get_models
from ts_benchmark.recording import save_log
from ts_benchmark.utils.parallel import ParallelBackend
from datetime import datetime
import os
import json
from collections import OrderedDict
import openpyxl

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    # the possible values of the meta-info field 'size'
    size_value: List
    # the class of data source for this dataset
    datasrc_class: Type[DataSource]


PREDEFINED_DATASETS = {
    "large_forecast": DatasetInfo(
        size_value=["large", "small"],
        datasrc_class=LocalForecastingDataSource,
    ),
    "small_forecast": DatasetInfo(size_value=["small"], datasrc_class=LocalForecastingDataSource),
    "user_forecast": DatasetInfo(size_value=["user"], datasrc_class=LocalForecastingDataSource),
}


def save_hyperparams_to_excel(MSE, MAE, RMSE, inf_time, args_info, model_info, time_info):
    """
    保存 MSE/MAE/inf_time、超参数、model_info、time_info 到 Excel 文件
    其中 MSE、MAE、inf_time 设置小数点格式，其他列正常显示。
    """
    # 模型名
    model_full = args_info["model_name"][0]
    model_name = model_full.rsplit(".", 1)[-1]
    # 数据集名
    data_full = args_info["data_name_list"][0]
    data_name = data_full.rsplit(".", 1)[0]
    # 超参数
    hyper_params_str = args_info["model_hyper_params"][0]
    hyper_params = json.loads(hyper_params_str)

    # 把 MSE/MAE/inf_time 插到最前
    ordered_params = OrderedDict()
    ordered_params["MSE"] = float(MSE)
    ordered_params["MAE"] = float(MAE)
    ordered_params["RMSE"] = float(RMSE)
    ordered_params["Inference_time"] = float(inf_time)

    def to_float_if_possible(v):
        # 如果是 int 或 float，直接转 float
        if isinstance(v, (int, float)):
            return float(v)
        # 如果是字符串且能转成 float
        if isinstance(v, str):
            if v.replace(".", "", 1).replace("-", "", 1).isdigit():
                return float(v)
        # 其他情况原样返回
        return v

    all_dicts = [hyper_params, model_info, time_info]
    for d in all_dicts:
        for k, v in d.items():
            ordered_params[k] = to_float_if_possible(v)

    # 保存路径
    save_dir = os.path.join("result", "excel", data_name)
    os.makedirs(save_dir, exist_ok=True)
    excel_path = os.path.join(save_dir, f"{model_name}.xlsx")

    # 转 DataFrame
    new_row = pd.DataFrame([ordered_params])

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        # 补齐列
        for col in new_row.columns:
            if col not in df.columns:
                df[col] = None
        for col in df.columns:
            if col not in new_row.columns:
                new_row[col] = None
        # 保证列顺序：MSE, MAE, Inference_time 在最前
        cols = ["MSE", "MAE", "RMSE", "Inference_time"] + [
            c for c in df.columns if c not in ["MSE", "MAE", "RMSE", "Inference_time"]
        ]
        df = df[cols]
        new_row = new_row[cols]
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    # 先保存 Excel
    df.to_excel(excel_path, index=False)

    # 用 openpyxl 设置 MSE/MAE/Inference_time 显示小数格式
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    header = [cell.value for cell in ws[1]]
    for col_name in ["MSE", "MAE", "RMSE", "Inference_time", "lr"]:
        if col_name in header:
            col_idx = header.index(col_name) + 1
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=col_idx)
                if isinstance(cell.value, (int, float)):
                    cell.number_format = "0.000000"  # 保留 6 位小数

    wb.save(excel_path)


def filter_data(metadata: pd.DataFrame, size_value: List[str], feature_dict: Optional[Dict] = None) -> List[str]:
    """
    Filters the dataset based on given filters

    :param metadata: The meta information DataFrame.
    :param size_value: The allowed values of the 'size' meta-info field.
    :param feature_dict: A dictionary of filters where each key is a meta-info field
        and the corresponding value is the field value to keep. If None is given,
        no extra filter is applied.
    :return:
    """
    # Remove items with a value of None in feature_dict
    feature_dict = {k: v for k, v in feature_dict.items() if v is not None}
    # Use the reduce and and_ functions to filter data file names that meet the criteria
    filt_metadata = metadata
    if feature_dict is not None:
        filt_metadata = metadata[reduce(and_, (metadata[k] == v for k, v in feature_dict.items()))]
    filt_metadata = filt_metadata[filt_metadata["size"].isin(size_value)]
    return filt_metadata["file_name"].tolist()


def _get_model_names(model_names: List[str]):
    """
    Rename models if there exists duplications.

    If a model A appears multiple times in the list, each appearance will be renamed to
    `A`, `A_1`, `A_2`, ...

    :param model_names: A list of model names.
    :return: The renamed list of model names.
    """
    s = pd.Series(model_names)
    cumulative_counts = s.groupby(s).cumcount()
    return [f"{model_name}_{cnt}" if cnt > 0 else model_name for model_name, cnt in zip(model_names, cumulative_counts)]


def pipeline(
    data_config: dict,
    model_config: dict,
    evaluation_config: dict,
) -> List[str]:
    """
    Execute the benchmark pipeline process

    The pipline includes loading data, building models, evaluating models, and generating reports.

    :param data_config: Configuration for data loading.
    :param model_config: Configuration for model construction.
    :param evaluation_config: Configuration for model evaluation.
    """
    # prepare data
    # TODO: move these code into the data module, after the pipeline interface is unified
    dataset_name_list = data_config.get("data_set_name", ["small_forecast"])
    if not dataset_name_list:
        dataset_name_list = ["small_forecast"]
    if isinstance(dataset_name_list, str):
        dataset_name_list = [dataset_name_list]
    for dataset_name in dataset_name_list:
        if dataset_name not in PREDEFINED_DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name}.")

    data_src_type = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class
    if not all(PREDEFINED_DATASETS[dataset_name].datasrc_class is data_src_type for dataset_name in dataset_name_list):
        raise ValueError("Not supporting different types of data sources.")

    data_src: DataSource = PREDEFINED_DATASETS[dataset_name_list[0]].datasrc_class()
    data_name_list = data_config.get("data_name_list", None)
    if not data_name_list:
        data_name_list = []
        for dataset_name in dataset_name_list:
            size_value = PREDEFINED_DATASETS[dataset_name].size_value
            feature_dict = data_config.get("feature_dict", None)
            data_name_list.extend(filter_data(data_src.dataset.metadata, size_value, feature_dict=feature_dict))
    data_name_list = list(set(data_name_list))
    if not data_name_list:
        raise ValueError("No dataset specified.")
    data_src.load_series_list(data_name_list)
    data_server = GlobalStorageDataServer(data_src, ParallelBackend())
    data_server.start_async()
    # modeling
    model_factory_list = get_models(model_config)
    result_list = [eval_model(model_factory, data_name_list, evaluation_config) for model_factory in model_factory_list]
    model_save_names = [
        it.split(".")[-1] for it in _get_model_names([model_factory.model_name for model_factory in model_factory_list])
    ]
    log_file_names = []
    # for model_factory, result_itr, model_save_name in zip(model_factory_list, result_list, model_save_names):
    #     for i, result_df in enumerate(result_itr.collect()):
    #         log_file_names.append(
    #             save_log(
    #                 result_df,
    #                 evaluation_config["save_path"],
    #                 model_save_name if i == 0 else f"{model_save_name}-{i}",
    #             )
    #         )
    ## add
    result_itr = result_list[0].collect()
    result_df = list(result_itr)
    MSE = round(result_df[0]["mse_norm"].values[0], 10)
    MAE = round(result_df[0]["mae_norm"].values[0], 10)
    RMSE = round(result_df[0]["rmse_norm"].values[0], 10)
    inf_time = round(result_df[0]["inference_time"].values[0], 10)
    logger.info(f"MSE is {MSE}")
    logger.info(f"MAE is {MAE}")
    logger.info(f"RMSE is {RMSE}")
    # 保存为excel
    save_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)
    # 需要加载的文件
    json_files = {
        "args_info": "args.json",
        "model_info": "model_info.json",
        "time_info": "time.json",
    }
    # 用字典存结果
    loaded_data = {}
    for key, filename in json_files.items():
        path = os.path.join(save_dir, filename)
        with open(path, "r") as f:
            loaded_data[key] = json.load(f)
    # 分别取出
    args_info = loaded_data["args_info"]
    model_info = loaded_data["model_info"]
    time_info = loaded_data["time_info"]
    save_hyperparams_to_excel(MSE, MAE, RMSE, inf_time, args_info, model_info, time_info)
    return log_file_names
