###############################################################################
#
#  The MIT License (MIT)
#  Copyright (c) 2023 Philippe Ostiguy
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################

import re
import pandas as pd
from collections import deque
from ratelimiter import RateLimiter
from datetime import datetime, timedelta
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
torch.set_default_dtype(torch.float32)
from typing import Optional, List, Tuple, Union, Dict, Any, Callable
from dotenv import load_dotenv
from copy import deepcopy
from abc import ABC, abstractmethod
from pmaw import PushshiftAPI
import time
import pytz
from statsmodels.tsa.stattools import adfuller
import csv
import requests
import logging
from contextlib import suppress
import shutil
from darts import TimeSeries
import matplotlib.pyplot as plt
import scipy.stats as stats
load_dotenv()

from mean_reversion.config.config_utils import ConfigManager

from mean_reversion.utils import (
    read_json,
    string_to_datetime,
    est_dt_to_epoch,
    get_previous_market_date,
    create_file_if_not_exist,
    obtain_market_dates,
    read_csv_to_pd_formatted,
    write_to_csv_formatted,
    write_pd_to_csv
)
from mean_reversion.config.constants import (
    RAW_ATTRIBUTES,
    MODEL_PHASES,
    MODEL_DATA_PATH,
    ENGINEERED_DATA_TO_REMOVE,
    DATASETS
)


class DataProcessorHelper:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager.config
        self._market_dates = obtain_market_dates(
            start_date=self._config["common"]["start_date"],
            end_date=self._config["common"]["end_date"],
        )
        self._obtain_split_index_set()

    @property
    def market_dates(self) -> pd.DataFrame:
        return self._market_dates

    @property
    def data_to_write_ts(self) -> pd.DataFrame:
        return self._data_to_write_ts

    @data_to_write_ts.setter
    def data_to_write_ts(self, value: pd.DataFrame) -> None:
        self._data_to_write_ts = value

    def remove_first_transformed_data(
        self, data_to_process: pd.DataFrame
    ) -> pd.DataFrame:
        return data_to_process.iloc[ENGINEERED_DATA_TO_REMOVE:]

    def obtain_train_data(self, full_data_set: pd.DataFrame) -> pd.DataFrame:
        return full_data_set.loc[: self._train_split_index]

    def obtain_predict_data(self, full_data_set: pd.DataFrame) -> pd.DataFrame:
        return full_data_set.loc[self._train_split_index + 1 : self._predict_split_index]

    def obtain_test_data(self, full_data_set: pd.DataFrame) -> pd.DataFrame:
        return full_data_set.loc[self._predict_split_index + 1:]

    def _obtain_split_index_set(self) -> None:
        total_length = len(self._market_dates)
        self._train_split_index = int(total_length * self._config["common"]["train_test_split"][0])
        self._predict_split_index = self._train_split_index + int(total_length * self._config["common"]["train_test_split"][1])

    def write_ts_to_csv(
        self, path_write_file: Dict, columns_to_drop: Optional[List] = None
    ) -> None:
        self._convert_to_ts_data(columns_to_drop)
        self._write_ts_to_the_csv(path_write_file)

    def _write_ts_to_the_csv(self, path_to_write_file: Dict) -> None:
        write_to_csv_formatted(
            self._train_time_series, path_to_write_file["train"]
        )

        write_to_csv_formatted(
            self._predict_time_series, path_to_write_file["predict"]
        )
        write_to_csv_formatted(
            self._test_time_series, path_to_write_file["test"]
        )

    def _convert_to_ts_data(self, columns_to_drop: Optional[List]) -> None:
        if columns_to_drop:
            self._data_to_write_ts = self._data_to_write_ts.drop(
                columns=columns_to_drop
            )

        self._train_time_series = TimeSeries.from_dataframe(
            self.obtain_train_data(self._data_to_write_ts), freq="B"
        ).astype(np.float32)
        self._predict_time_series = TimeSeries.from_dataframe(
            self.obtain_predict_data(self._data_to_write_ts), freq="B"
        ).astype(np.float32)
        self._test_time_series = TimeSeries.from_dataframe(
            self.obtain_test_data(self._data_to_write_ts), freq="B"
        ).astype(np.float32)


class DataForModelSelector:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager.config
        self._data_for_model_train = None
        self._data_for_model_predict = None
        self._data_for_model_test = None
        self._datasets = DATASETS

    def run(self) -> None:
        last_data = {}
        for dataset in self._datasets:
            for sources in self._config["output"]:
                for data in sources['data']:
                    if not os.path.exists(data["engineered"][dataset]):
                        continue
                    if not hasattr(self, '_all_output'):
                        self._all_output = read_csv_to_pd_formatted(data["engineered"][dataset], sort_by_column_name='time')
                        last_data = data
                        continue
                    new_data = read_csv_to_pd_formatted(data["engineered"][dataset], 'time')
                    self._all_output= self._all_output.merge(new_data, on='time', how='outer')
                    last_data = data
            if not last_data:
                raise ValueError('No output file found')
            write_to_csv_formatted(
                self._all_output,
                last_data["model_data"][dataset],
                'time')
            del self._all_output


        self._output_data_train = read_csv_to_pd_formatted(
            last_data["model_data"]['train'], "time"
        )

        for input_in_config in self._config["inputs"]["past_covariates"][
            "sources"
        ]:
            for current_input in input_in_config["data"]:
                train_file_path = current_input["engineered"]["train"]
                predict_file_path = current_input["engineered"]["predict"]
                test_file_path = current_input["engineered"]["test"]
                if not os.path.exists(train_file_path):
                    continue
                asset_name = current_input["asset"]
                data_train = self._load_and_prefix_data(
                    train_file_path, asset_name, ["time"]
                )

                data_predict = self._load_and_prefix_data(
                    predict_file_path, asset_name, ["time"]
                )

                data_test = self._load_and_prefix_data(
                    test_file_path, asset_name, ["time"]
                )

                if not self._output_data_train["time"].equals(
                    data_train["time"]
                ):
                    raise ValueError(
                        f"time values are not consistent in {asset_name}"
                    )

                if self._data_for_model_train is None:
                    self._data_for_model_train = data_train
                    self._data_for_model_predict = data_predict
                    self._data_for_model_test = data_test
                else:
                    duplicated_columns = (
                        self._data_for_model_train.columns.intersection(
                            data_train.columns
                        )
                    )
                    duplicated_columns = [
                        col for col in duplicated_columns if col != "time"
                    ]
                    if duplicated_columns:
                        raise ValueError(
                            f"Duplicate columns found for asset {asset_name}: {', '.join(duplicated_columns)}"
                        )

                    self._data_for_model_train = pd.merge(
                        self._data_for_model_train,
                        data_train,
                        on="time",
                        how="left",
                    )
                    self._data_for_model_predict = pd.merge(
                        self._data_for_model_predict,
                        data_predict,
                        on="time",
                        how="left",
                    )
                    self._data_for_model_test = pd.merge(
                        self._data_for_model_test,
                        data_test,
                        on="time",
                        how="left",
                    )

        if self._config["common"]["model_phase"] == "train":
            self._make_correlated_feature_removal()
        else:
            self._apply_correlated_feature_removal()

        write_to_csv_formatted(
            self._data_for_model_train,
            self._config["inputs"]["past_covariates"]["common"]["model_data"][
                "train"
            ],
            'time'
        )
        write_to_csv_formatted(
            self._data_for_model_predict,
            self._config["inputs"]["past_covariates"]["common"]["model_data"][
                "predict"
            ],
            'time'
        )
        write_to_csv_formatted(
            self._data_for_model_test,
            self._config["inputs"]["past_covariates"]["common"]["model_data"][
                "test"
            ],
            'time'
        )

    def _make_correlated_feature_removal(self) -> None:
        corr_matrix = self._data_for_model_train.corr().abs()
        corr_sum = pd.DataFrame(
            {"sum": corr_matrix.sum(), "col": corr_matrix.columns}
        )
        corr_sum_sorted = corr_sum.sort_values("sum", ascending=False)

        to_keep = ["time"]

        for _, row in corr_sum_sorted.iterrows():
            column = row["col"]
            if column == "time":
                continue
            if all(
                corr_matrix[column][to_keep]
                <= self._config["common"]["correlation_threshold"]
            ):
                to_keep.append(column)

        with open(
            self._config["inputs"]["past_covariates"]["common"]["pickle"][
                "features_to_keep"
            ],
            "wb",
        ) as f:
            pickle.dump(to_keep, f)

        self._raise_error_if_missing_columns(
            to_keep, self._data_for_model_train, "training data"
        )
        self._raise_error_if_missing_columns(
            to_keep, self._data_for_model_predict, "prediction data"
        )
        self._raise_error_if_missing_columns(
            to_keep, self._data_for_model_test, "test data"
        )

        self._data_for_model_train = self._data_for_model_train[to_keep]
        self._data_for_model_predict = self._data_for_model_predict[to_keep]
        self._data_for_model_test= self._data_for_model_test[to_keep]

    def _apply_correlated_feature_removal(self) -> None:
        with open(
            self._config["inputs"]["past_covariates"]["common"]["pickle"][
                "features_to_keep"
            ],
            "rb",
        ) as f:
            to_keep = pickle.load(f)
        self._raise_error_if_missing_columns(
            to_keep, self._data_for_model_predict, "prediction data"
        )
        self._raise_error_if_missing_columns(
            to_keep, self._data_for_model_test, "test data"
        )
        self._data_for_model_predict = self._data_for_model_predict[to_keep]
        self._data_for_model_train = self._data_for_model_train[to_keep]
        self._data_for_model_test = self._data_for_model_test[to_keep]
    @staticmethod
    def _load_and_prefix_data(
        file_path: str,
        asset_name: str,
        columns_not_to_change: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        if columns_not_to_change is not None:
            columns_to_change = [
                col for col in data.columns if col not in columns_not_to_change
            ]
            data_to_change = data[columns_to_change].add_prefix(
                f"{asset_name}_"
            )
            data = pd.concat(
                [data_to_change, data[columns_not_to_change]], axis=1
            )
        else:
            data = data.add_prefix(f"{asset_name}_")
        return data

    @staticmethod
    def _raise_error_if_missing_columns(
        columns_to_keep: List[str], data: pd.DataFrame, data_name: str
    ) -> None:
        missing_columns = set(columns_to_keep) - set(data.columns)
        if missing_columns:
            raise KeyError(
                f"Columns not found in {data_name}: {missing_columns}"
            )


class BaseInputOutputDataEngineering(ABC):
    def __init__(
        self,
        processor: "BaseDataProcessor",
        data_processor_helper: DataProcessorHelper,
        columns_to_drop_ts: Optional[List] = RAW_ATTRIBUTES[0],
    ):
        self._processor = processor
        self._data_processor_helper = data_processor_helper
        self._engineered_data: pd.DataFrame = pd.DataFrame()
        self._columns_to_drop_ts = columns_to_drop_ts
        self._scaler_to_apply = StandardScaler

    @abstractmethod
    def run_feature_engineering(self):
        pass


class InputDataEngineering(BaseInputOutputDataEngineering):
    def __init__(
        self,
        processor: "BaseDataProcessor",
        data_processor_helper: DataProcessorHelper,
        data_for_stationarity_to_remove: Optional[
            int
        ] = ENGINEERED_DATA_TO_REMOVE,
    ):
        super().__init__(processor, data_processor_helper)

        self._asset_name: str = ""
        self._current_column: str = ""
        self._data_for_stationarity_to_remove = data_for_stationarity_to_remove
        self._transformations: List[
            Tuple[str, Callable[[pd.Series], Optional[pd.Series]]]
        ] = [
            ("identity", self._make_identity),
            ("first_difference", self._make_first_difference),
            ("logarithmic", self._make_logarithmic),
            ("ratio", self._make_ratio),
        ]

        self._inference_transformations: List[
            Tuple[str, Callable[[pd.Series], pd.Series]]
        ] = [
            ("identity", self._apply_identity),
            ("ratio", self._apply_ratio),
            ("first_difference", self._apply_first_difference),
            ("logarithmic", self._apply_logarithmic),

        ]

        self._stationarity_applied = {}
        self._scaler_applied = {}

    def _handle_missing_inf(self, data: pd.Series) -> Optional[pd.Series]:
        missing_or_inf = data.isna() | np.isinf(data)
        missing_or_inf_count = missing_or_inf.sum()
        missing_indices = set(range(1, len(data) + 1)) - set(data.index)

        if (missing_or_inf_count + len(missing_indices)) / len(
            data
        ) > self._processor._config["common"]["max_missing_data"]:
            return None

        return self._handle_missing_inf_apply(data)

    def _handle_missing_inf_apply(self, data: pd.Series) -> pd.Series:
        missing_indices = set(range(1, len(data))) - set(data.index)
        adjusted_data = data.copy()

        if not 0 in missing_indices:
            for index in missing_indices:
                prev_index = max(i for i in adjusted_data.index if i < index)
                adjusted_data = adjusted_data.reindex(range(1, index)).fillna(
                    adjusted_data[prev_index]
                )

        first_valid_index = adjusted_data.first_valid_index()
        adjusted_data.iloc[
            : self._data_for_stationarity_to_remove
        ] = adjusted_data[first_valid_index]

        adjusted_data = adjusted_data.sort_index()
        adjusted_data = adjusted_data.replace([np.inf, -np.inf], np.nan)
        adjusted_data = adjusted_data.fillna(method="ffill")

        return adjusted_data

    def _make_identity(self, data: pd.Series) -> Optional[pd.Series]:
        data = self._handle_missing_inf(data)
        return data

    def _apply_identity(self, data: pd.Series) -> pd.Series:
        data = self._handle_missing_inf_apply(data)
        return data

    def _make_ratio(self, data: pd.Series) -> Optional[pd.Series]:
        ratio_data = data / data.shift(1)
        ratio_data = self._handle_missing_inf(ratio_data)
        return ratio_data

    def _apply_ratio(self, data: pd.Series) -> pd.Series:
        ratio_data = data / data.shift(1)
        ratio_data = self._handle_missing_inf_apply(ratio_data)
        return ratio_data

    def _make_logarithmic(self, data: pd.Series) -> Optional[pd.Series]:
        data = np.log(data + 1e-7)
        data = self._handle_missing_inf(data)
        return data

    def _apply_logarithmic(self, data: pd.Series) -> pd.Series:
        data = np.log(data + 1e-7)
        data = self._handle_missing_inf_apply(data)
        return data

    def _make_first_difference(self, data: pd.Series) -> Optional[pd.Series]:
        data_diff = data.diff()
        data_diff = self._handle_missing_inf(data_diff)
        return data_diff

    def _apply_first_difference(self, data: pd.Series) -> pd.Series:
        data_diff = data.diff()
        data_diff = self._handle_missing_inf_apply(data_diff)
        return data_diff

    def _obtain_adf_p_value(self, time_series: pd.Series) -> float:
        adf_result = adfuller(time_series)
        return adf_result[1]

    def coordinate_stationarity(self, data: pd.Series) -> Optional[str]:
        valid_transformations = []
        if not self._processor._config["common"]["make_data_stationary"] :
            return "identity"

        for name, transform_function in self._transformations:
            transformed_data = transform_function(data)
            if transformed_data is None or (
                    transformed_data.isna() | np.isinf(transformed_data)).any():
                return None

            if len(set(transformed_data)) == 1:
                return None

            p_value = self._obtain_adf_p_value(transformed_data)
            if p_value < 0.05:
                skewness = stats.skew(transformed_data)
                kurtosis = stats.kurtosis(transformed_data)
                if not self._processor._config["common"]["check_bell_shape"]:
                    valid_transformations.append(
                        (name, p_value, skewness, kurtosis))

                elif -2 <= skewness <= 7 and -2 <= kurtosis <= 7:
                    valid_transformations.append(
                        (name, p_value, skewness, kurtosis))
        #self.TEMPO_FCT_MAKE_PLOT(data)
        if not valid_transformations:
            logging.warning(
                f"Asset {self._asset_name} and column '{self._current_column}' "
                f"could not be made stationary with the given transformations."
            )
            return None

        for transformation in valid_transformations:
            if transformation[0] == "identity":
                return "identity"
        final_transformation = min(valid_transformations, key=lambda x: (x[3], x[2]))[0]
        return final_transformation

    def TEMPO_FCT_MAKE_PLOT(self,data):
        log_data = self._make_logarithmic(data)
        first_diff = self._make_first_difference(data)
        ratio_data = self._make_ratio(data)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(data)
        plt.title('Original Series')
        plt.xlabel('Index')
        plt.ylabel('Value')

        plt.subplot(2, 2, 2)
        plt.plot(log_data)
        plt.title('Logarithm of Series')
        plt.xlabel('Index')
        plt.ylabel('Log(Value)')

        plt.subplot(2, 2, 3)
        plt.plot(first_diff)
        plt.title('First Difference of Series')
        plt.xlabel('Index')
        plt.ylabel('First Difference')

        plt.subplot(2, 2, 4)
        plt.plot(ratio_data)
        plt.title('Ratio of Series')
        plt.xlabel('Index')
        plt.ylabel('Ratio')

        plt.tight_layout()
        plt.show(block=False)
        plt.close()

    def run_feature_engineering(self):
        self._engineered_data = read_csv_to_pd_formatted(
            self._processor._config["data"][
                self._processor._current_data_index
            ]["preprocessed"]
        )
        self._columns_to_transform = [
            column
            for column in self._engineered_data.columns
            if column != RAW_ATTRIBUTES[0]
        ]
        self._coordinate_engineering_methods()
        self._dispatch_feature_engineering("data_stationary")
        self._engineered_data = (
            self._data_processor_helper.remove_first_transformed_data(
                self._engineered_data
            )
        )
        self._transformed_columns = [ col for col in
                                      self._engineered_data.columns.tolist()
                                      if col != RAW_ATTRIBUTES[0]]

        if self._transformed_columns :
            if self._processor._config["common"]["scaling"]:
                self._dispatch_feature_engineering("data_scaling")
            self._data_processor_helper.data_to_write_ts = self._engineered_data

            self._data_processor_helper.write_ts_to_csv(
                self._processor._config["data"][
                    self._processor._current_data_index
                ]["engineered"],
                self._columns_to_drop_ts,
            )

            self._perform_pickle_operation(
                self._processor._config["common"]["pickle"][
                    "transformations_for_scaler"
                ],
                "wb",
                self._scaler_applied,
            )
            self._perform_pickle_operation(
                self._processor._config["common"]["pickle"][
                    "transformations_for_stationarity"
                ],
                "wb",
                self._stationarity_applied,
            )

    def _dispatch_feature_engineering(self, operation):
        if (
            self._processor._config["common"]["model_phase"]
            == self._processor._model_phase[0]
        ):
            getattr(self, f"_make_{operation}")()
        else:
            getattr(self, f"_apply_{operation}")()

    def _obtain_pickle_inference(self, file: str) -> Dict[str, Any]:
        dictionary = self._perform_pickle_operation(file, "rb")
        return dictionary.get(
            self._processor._config["data"][
                self._processor._current_data_index
            ]["asset"]
        )

    def _perform_pickle_operation(
        self,
        filepath: str,
        operation: str,
        transformation: Optional[Union[Dict, StandardScaler]] = None,
    ) -> Any:
        with open(filepath, operation) as file:
            if operation == "rb":
                with suppress(
                    (
                        pickle.PicklingError,
                        pickle.UnpicklingError,
                        FileNotFoundError,
                        Exception,
                    )
                ):
                    return pickle.load(file)
            elif operation == "wb":
                assert (
                    transformation is not None
                ), "Cannot write None transformation"
                with suppress(
                    (
                        pickle.PicklingError,
                        pickle.UnpicklingError,
                        FileNotFoundError,
                        Exception,
                    )
                ):
                    try:
                        existing_data = pickle.load(file)
                    except (
                        pickle.PicklingError,
                        pickle.UnpicklingError,
                        FileNotFoundError,
                        Exception,
                    ):
                        existing_data = {}

                    existing_data.update(transformation)
                    pickle.dump(existing_data, file)

    def _make_data_scaling(self) -> None:
        self._scaler_applied[
            self._processor._config["data"][
                self._processor._current_data_index
            ]["asset"]
        ] = {}
        train_data = self._data_processor_helper.obtain_train_data(
            self._engineered_data
        ).copy()
        predict_data = self._data_processor_helper.obtain_predict_data(
            self._engineered_data
        ).copy()
        test_data = self._data_processor_helper.obtain_test_data(
            self._engineered_data
        ).copy()

        for column in self._transformed_columns:
            scaler = self._scaler_to_apply()
            train_data[column] = scaler.fit_transform(train_data[[column]])
            predict_data[column] = scaler.transform(predict_data[[column]])
            test_data[column] = scaler.transform(test_data[[column]])
            self._scaler_applied[
                self._processor._config["data"][
                    self._processor._current_data_index
                ]["asset"]
            ][column] = scaler

        self._engineered_data = pd.concat([train_data, predict_data, test_data])

    def _perform_common_not_stationary(self, column: str) -> None:
        self._engineered_data = self._engineered_data.drop(column, axis=1)


    def _apply_data_scaling(self) -> None:
        self._scaler_applied = self._obtain_pickle_inference(
            self._processor._config["common"]["pickle"][
                "transformations_for_scaler"
            ]
        )

        if self._scaler_applied is not None:
            for column in self._transformed_columns:
                if column in self._scaler_applied:
                    self._engineered_data[column] = self._scaler_applied[
                        column
                    ].transform(self._engineered_data[[column]])

    def _apply_data_stationary(self) -> None:
        self._stationarity_applied = self._obtain_pickle_inference(
            self._processor._config["common"]["pickle"][
                "transformations_for_stationarity"
            ]
        )

        if self._stationarity_applied is not None:
            for column in self._columns_to_transform:
                transformation_name = self._stationarity_applied[column]
                if transformation_name == "not stationary":
                    self._perform_common_not_stationary(column)
                else:
                    transformation_func_dict = dict(
                        self._inference_transformations
                    )
                    transformation_func = transformation_func_dict[
                        transformation_name
                    ]
                    self._engineered_data[column] = transformation_func(
                        self._engineered_data[column]
                    )

    def _make_data_stationary(self) -> None:
        self._asset_name = self._processor._config["data"][
            self._processor._current_data_index
        ]["asset"]
        self._stationarity_applied[
            self._processor._config["data"][
                self._processor._current_data_index
            ]["asset"]
        ] = {}

        train_data = self._data_processor_helper.obtain_train_data(
            self._engineered_data
        )
        predict_data = self._data_processor_helper.obtain_predict_data(
            self._engineered_data
        )
        test_data = self._data_processor_helper.obtain_test_data(
            self._engineered_data
        )

        for column in self._columns_to_transform:
            self._current_column = column
            best_transformation = self.coordinate_stationarity(
                train_data[column]
            )

            if best_transformation is None:
                self._perform_common_not_stationary(column)
                self._stationarity_applied[
                    self._processor._config["data"][
                        self._processor._current_data_index
                    ]["asset"]
                ][column] = "not stationary"
            else:
                self._stationarity_applied[
                    self._processor._config["data"][
                        self._processor._current_data_index
                    ]["asset"]
                ][column] = best_transformation
                self._engineered_data[column] = dict(self._transformations)[
                    best_transformation
                ](pd.concat([train_data[column], predict_data[column], test_data[column]]))

    def _coordinate_engineering_methods(self):
        pass


class OutputDataEngineering(BaseInputOutputDataEngineering):
    def __init__(
        self,
        processor: "BaseDataProcessor",
        data_processor_helper: DataProcessorHelper,
        data_model_dir: Optional[str] = MODEL_DATA_PATH,
    ):
        super().__init__(processor, data_processor_helper)
        self._data_model_dir = data_model_dir

    def run_feature_engineering(self):
        self._engineered_data = read_csv_to_pd_formatted(
            self._processor._config["data"][
                self._processor._current_data_index
            ]["preprocessed"]
        )

        target_column = (self._processor._config["data"][self._processor._current_data_index]['asset'] + '_target').lower()
        if not self._processor._config["common"]["make_data_stationary"]:
            if '4. close' in self._engineered_data.columns:
                self._engineered_data[target_column] = self._engineered_data['4. close']
            elif 'value' in self._engineered_data.columns:
                self._engineered_data[target_column] = self._engineered_data['value']

            else :
                raise ValueError('Columns most have open and close name or value name')

        if '1. open' in self._engineered_data.columns and '4. close' in self._engineered_data.columns:
            self._engineered_data[target_column] = (
            self._engineered_data['4. close']
            / self._engineered_data["1. open"]
        ) - 1

        elif 'value' in self._engineered_data.columns :
            self._engineered_data[target_column] \
                = (self._engineered_data['value']/
                   self._engineered_data["value"].shift(1)) - 1
            self._engineered_data.iloc[0] = 0
        else :
            raise ValueError('Columns most have open and close name or value name')

        self._engineered_data =\
            self._data_processor_helper.remove_first_transformed_data(
                self._engineered_data
            )
        columns_to_keep = {target_column}
        self._engineered_data = self._engineered_data[[*columns_to_keep]]

        train_data = self._data_processor_helper.obtain_train_data(
            self._engineered_data
        ).copy()
        predict_data = self._data_processor_helper.obtain_predict_data(
            self._engineered_data
        ).copy()

        test_data = self._data_processor_helper.obtain_test_data(
            self._engineered_data
        ).copy()

        self._data_processor_helper.data_to_write_ts = \
            pd.concat([train_data[list(columns_to_keep)[0]],
                       predict_data[list(columns_to_keep)[0]],
                       test_data[list(columns_to_keep)[0]]]).to_frame()
        self._data_processor_helper.write_ts_to_csv(
            self._processor._config["data"][
                self._processor._current_data_index
            ]["engineered"]
        )


class BaseDataProcessor(ABC):
    def __init__(
        self,
        config: dict,
        data_processor_helper: DataProcessorHelper,
        is_input_feature: Optional[bool] = True,
        model_phase: Optional[str] = MODEL_PHASES,
        **kwargs,
    ):
        super().__init__()
        self._config = config
        self._model_phase = model_phase
        self._is_current_asset_dropped = False
        self._attributes_to_delete = read_json(
            "resources/configs/models_support.json"
        )["attributes_to_discard"]
        if is_input_feature:
            pass
            self._feature_engineering_strategy = InputDataEngineering(
                self, data_processor_helper
            )
        else:
            self._feature_engineering_strategy = OutputDataEngineering(
                self, data_processor_helper
            )


    def _run_common_fetch(self):
        start_date = self._config["common"]["start_date"]
        end_date = self._config["common"]["end_date"]
        write_new_data = False
        try :
            self._raw_data = read_csv_to_pd_formatted(self._config["data"][self._current_data_index]["raw"])
        except FileNotFoundError :
            self._raw_data = self._run_fetch()
            write_pd_to_csv(data=self._raw_data,
                            file=self._config["data"][self._current_data_index][
                                "raw"])
            return

        pd_datetime = pd.to_datetime(self._raw_data['date'])
        first_date = pd_datetime.iloc[0]
        last_date = pd_datetime.iloc[-1]


        if first_date > pd.to_datetime(self._config["common"]["start_date"]):
            self._config["common"]["end_date"] = (first_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            self._raw_data = pd.concat([self._run_fetch(),self._raw_data],ignore_index=True)
            write_new_data = True
        if last_date < pd.to_datetime(self._config["common"]["end_date"]):
            self._config["common"]["start_date"] = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            self._raw_data = pd.concat([self._run_fetch(), self._raw_data],
                                       ignore_index=True)
            write_new_data = True

        if write_new_data :
            self._raw_data.drop_duplicates(subset=[RAW_ATTRIBUTES[0]], keep='first', inplace=True)
            write_pd_to_csv(data=self._raw_data,file=self._config["data"][self._current_data_index]["raw"])
            self._config["common"]["start_date"] = start_date
            self._config["common"]["end_date"] = end_date


    @abstractmethod
    def _run_fetch(self) -> pd.DataFrame:
        pass


    def run(self):
        assets_to_remove = []
        for index, _ in enumerate(self._config["data"]):
            self._current_data_index = index
            self._is_current_asset_dropped = False

            self._execute_if_not_dropped(self._run_common_fetch,
                condition=self._config["common"]["fetching"]
            )
            self._execute_if_not_dropped(
                self._run_preprocess,
                condition=self._config["common"]["preprocessing"]
            )
            self._execute_if_not_dropped(
                self._feature_engineering_strategy.run_feature_engineering,
                condition=self._config["common"]["engineering"]
            )
            if self._is_current_asset_dropped:
                assets_to_remove.append(index)

        for index in sorted(assets_to_remove, reverse=True):
            del self._config["data"][index]

    def _execute_if_not_dropped(
        self, method: Callable, *args, condition: Optional[bool] =True, **kwargs
    ) -> None:
        if condition and not self._is_current_asset_dropped:
            method(*args,**kwargs)

    def _run_preprocess(self):
        self._preprocessed_data = read_csv_to_pd_formatted(
            self._config["data"][self._current_data_index]["raw"]
        )
        self._preprocessed_data = self._preprocessed_data.rename(
            columns=str.lower
        )
        self._preprocessed_data = self._make_attribute_index(
            self._preprocessed_data
        )
        self._obtain_data_within_date()
        self._keep_trading_day()
        self._remove_attributes_in_data()
        self._clean_data()
        self._preprocessed_data = self._handle_missing_data(
            self._preprocessed_data
        )
        self._execute_if_not_dropped(write_to_csv_formatted,
            self._preprocessed_data,
            self._config["data"][self._current_data_index]["preprocessed"],
        )


    @staticmethod
    def _make_attribute_index(
        data: pd.DataFrame, attribute: Optional[str] = RAW_ATTRIBUTES[0]
    ) -> pd.DataFrame:
        return data.set_index(
            pd.to_datetime(data[attribute]), inplace=False, drop=False
        )

    def _remove_attributes_in_data(self) -> None:
        attributes_to_remove = [
            item.lower() for item in self._attributes_to_delete
        ]

        columns_to_drop = []
        for column in self._preprocessed_data.columns:
            for attribute_to_remove in attributes_to_remove:
                if attribute_to_remove in column.lower():
                    columns_to_drop.append(column)

        self._preprocessed_data = self._preprocessed_data.drop(
            columns=columns_to_drop
        )

    def _clean_data(self) -> None:
        mask = self._preprocessed_data.isin(["", ".", None])
        rows_to_remove = mask.any(axis=1)
        self._preprocessed_data = self._preprocessed_data.loc[~rows_to_remove]

    def _keep_trading_day(self) -> None:
        market_dates = obtain_market_dates(
            start_date=self._config["common"]["start_date"],
            end_date=self._config["common"]["end_date"],
        )
        self._preprocessed_data = self._preprocessed_data.loc[
            self._preprocessed_data.index.isin(market_dates.index)
        ]

    def _obtain_data_within_date(
        self, date_column: Optional[str] = RAW_ATTRIBUTES[0]
    ) -> None:
        self._preprocessed_data = self._preprocessed_data[
            (
                self._preprocessed_data[date_column]
                >= self._config["common"]["start_date"]
            )
            & (
                self._preprocessed_data[date_column]
                <= self._config["common"]["end_date"]
            )
        ]

    def _insert_missing_date(
        self, data: pd.DataFrame, date: str, column: str
    ) -> pd.DataFrame:
        if date not in data[column].values:
            prev_date = (
                data[data[column] < date].iloc[-1]
                if not data[data[column] < date].empty
                else data.iloc[0]
            )
            new_row = prev_date.copy()
            new_row[column] = date
            data = (
                pd.concat([data, new_row.to_frame().T], ignore_index=True)
                .sort_values(by=column)
                .reset_index(drop=True)
            )
        return data

    def _handle_missing_data(
        self,
        data: pd.DataFrame,
        threshold: Optional[int] = 1,
        column: Optional[str] = RAW_ATTRIBUTES[0],
    ) -> Union[None,pd.DataFrame]:
        modified_data = data.copy()
        market_open_dates = obtain_market_dates(
            start_date=self._config["common"]["start_date"],
            end_date=self._config["common"]["end_date"],
        )

        market_open_dates["count"] = 0
        market_open_dates.index = market_open_dates.index.strftime("%Y-%m-%d")
        date_counts = data[column].value_counts()

        market_open_dates["count"] = market_open_dates.index.map(
            date_counts
        ).fillna(0)

        missing_dates = market_open_dates.loc[
            market_open_dates["count"] < threshold
        ]

        if not missing_dates.empty:
            max_count = (
                len(market_open_dates)
                * self._config["common"]["max_missing_data"]
            )

            if len(missing_dates) > max_count:
                logging.warning(
                    f"For current asset {self._config['data'][self._current_data_index]['asset']},there are "
                    f"{len(missing_dates)} missing data which is than the maximum threshold of "
                    f"{self._config['common']['max_missing_data'] * 100}%"
                )
                self._drop_current_asset()
                return None
            else:
                for date, row in missing_dates.iterrows():
                    modified_data = self._insert_missing_date(
                        modified_data, date, column
                    )
        return modified_data

    def _drop_current_asset(self):
        self._is_current_asset_dropped = True

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        text = text.lower()
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        return re.sub(" +", " ", text)


class FutureCovariatesProcessor:
    def __init__(self, config_manager, data_processor_helper):
        self._config = config_manager.config
        self._data_processor_helper = data_processor_helper
        self._data_processor_helper.market_dates[
            RAW_ATTRIBUTES[0]
        ] = self._data_processor_helper.market_dates.index
        self._data_processor_helper.market_dates.index = pd.RangeIndex(
            start=0, stop=len(self._data_processor_helper.market_dates)
        )
        self._future_covariates = (
            self._data_processor_helper.remove_first_transformed_data(
                self._data_processor_helper.market_dates.copy()
            )
        )
        self._train_data = self._data_processor_helper.obtain_train_data(
            self._future_covariates.copy()
        )
        self._predict_data = self._data_processor_helper.obtain_predict_data(
            self._future_covariates.copy()
        )
        self._test_data = self._data_processor_helper.obtain_test_data(
            self._future_covariates.copy()
        )
        self._scaler = {}

    def run(self):
        self._train_data = self._add_covariates(
            self._train_data.copy(), fit=True
        )
        self._predict_data = self._add_covariates(
            self._predict_data.copy(), fit=False
        )
        self._test_data = self._add_covariates(
            self._test_data.copy(), fit=False
        )
        self._data_processor_helper.data_to_write_ts = pd.concat(
            [self._train_data, self._predict_data, self._test_data]
        )[self._config["inputs"]["future_covariates"]["data"]]
        self._data_processor_helper.write_ts_to_csv(
            self._config["inputs"]["future_covariates"]["common"]["model_data"]
        )

    def _add_covariates(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        covariates = self._config["inputs"]["future_covariates"]["data"]
        if covariates and "day" in covariates:
            df["day"] = df[RAW_ATTRIBUTES[0]].dt.day
            df["day"] = self._scale("day", df["day"], fit)
        if "month" in covariates:
            df["month"] = df[RAW_ATTRIBUTES[0]].dt.month
            df["month"] = self._scale("month", df["month"], fit)
        return df

    def _scale(self, feature: str, data: pd.Series, fit: bool) -> pd.Series:

        if not self._config["common"]["scaling"]:
            return data
        data_reshaped = data.values.reshape(-1, 1)
        if fit :
            self._scaler[feature] = MinMaxScaler()
            self._scaler[feature].fit(data_reshaped)
            return self._scaler[feature].transform(data_reshaped)
        else :
            return self._scaler[feature].transform(data_reshaped)

class Reddit(BaseDataProcessor):
    def __init__(
        self,
        specific_config,
        data_processor_helper,
        is_input_feature=True,
        **kwargs,
    ):
        super().__init__(
            specific_config, data_processor_helper, is_input_feature, **kwargs
        )
        self._start_date_unix = est_dt_to_epoch(
            string_to_datetime(self._config["common"]["start_date"]).replace(
                hour=9, minute=0, second=0
            )
        )
        end_date_dt = (
            string_to_datetime(self._config["common"]["end_date"])
            + timedelta(days=1)
        ).replace(hour=9, minute=0, second=0)
        self._current_date = self._config["common"]["end_date"]
        self._end_date_unix = est_dt_to_epoch(end_date_dt)
        self._comments = []
        self._push_shift = PushshiftAPI()

    def _run_fetch(self):
        while self._start_date_unix < self._end_date_unix:
            self._obtain_data()
            self._filter_raw_data()
            self._save_data_reddit()
            self._end_date_unix -= 86400
            self._end_date_unix = self._check_and_adjust_unix_timestamp(
                self._end_date_unix
            )
            dt = datetime.strptime(self._current_date, "%Y-%m-%d")
            new_dt = dt - timedelta(days=1)
            self._current_date = new_dt.strftime("%Y-%m-%d")

    def _run_preprocess(self):
        self._preprocessed_data = read_csv_to_pd_formatted(
            (self._config["data"][self._current_data_index]["raw"]),
            "created_utc",
        )
        self._obtain_data_within_date()
        self._keep_valid_market_data_reddit()
        self._clean_reddit_data()
        self._preprocessed_data = self._handle_missing_data(
            self._preprocessed_data,
            threshold=self._config["data"][self._current_data_index]["size"],
        )
        write_to_csv_formatted(
            self._preprocessed_data,
            self._config["data"][self._current_data_index]["preprocessed"],
        )

    def _keep_valid_market_data_reddit(self):
        self._preprocessed_data[RAW_ATTRIBUTES[0]] = pd.to_datetime(
            self._preprocessed_data[RAW_ATTRIBUTES[0]]
        )
        last_market_date = None
        self._preprocessed_data[RAW_ATTRIBUTES[0]], last_market_date = zip(
            *self._preprocessed_data[RAW_ATTRIBUTES[0]].apply(
                lambda x: get_previous_market_date(
                    x, last_market_date=last_market_date
                )
            )
        )
        self._preprocessed_data[RAW_ATTRIBUTES[0]] = self._preprocessed_data[
            RAW_ATTRIBUTES[0]
        ].apply(lambda x: x.strftime("%Y-%m-%d"))

    def _keep_unique_values(self, data) -> List[dict]:
        seen = set()
        unique_data = []
        for row in data:
            key = (row["created_utc"], row[RAW_ATTRIBUTES[1]])
            if key not in seen:
                seen.add(key)
                unique_data.append(row)
        return unique_data

    def _clean_reddit_data(self):
        self._preprocessed_data = self._preprocessed_data[
            self._preprocessed_data[RAW_ATTRIBUTES[1]].notnull()
            & (self._preprocessed_data[RAW_ATTRIBUTES[1]].str.strip() != "")
        ]
        self._preprocessed_data = self._preprocessed_data.drop_duplicates()
        self._preprocessed_data = self._preprocessed_data.drop_duplicates(
            subset=["created_utc", RAW_ATTRIBUTES[1]]
        )
        self._preprocessed_data[RAW_ATTRIBUTES[1]] = self._preprocessed_data[
            RAW_ATTRIBUTES[1]
        ].apply(self._clean_text)

    def _coordinate_engineering_methods(self):
        self._coordinate_roberta_engineer()

    def _coordinate_roberta_engineer(self):
        # IL FAUT RETRAVAILLER CETTE METHODE. IL NE FAUT PAS SAUVEGARGER LES DONNÉES
        # DANS UN CSV, MAIS UTILISER UN PANDAS
        self._tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )

        with open(
            self._config["data"][self._current_data_index]["engineered"],
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    RAW_ATTRIBUTES[0],
                    "negative",
                    "neutral",
                    "positive",
                ],
            )
            writer.writeheader()

            for date, group in self._engineered_data.groupby(RAW_ATTRIBUTES[0]):
                sentiment_prob_sum = [0.0, 0.0, 0.0]

                for index, text in enumerate(group[RAW_ATTRIBUTES[1]]):
                    utc = group["created_utc"].iloc[index]
                    try:
                        sentiment_probs = self._predict_sentiment_probabilities(
                            text
                        )
                        sentiment_prob_sum = [
                            x + y
                            for x, y in zip(sentiment_prob_sum, sentiment_probs)
                        ]
                    except Exception as e:
                        logging.error(
                            f"Error processing text: Date={date}, Value={text}, utc = {utc}"
                        )
                        continue

                average_sentiment_probs = [
                    x / len(group[RAW_ATTRIBUTES[1]])
                    for x in sentiment_prob_sum
                ]
                avg_sentiments = {
                    RAW_ATTRIBUTES[0]: date,
                    "negative": average_sentiment_probs[0],
                    "neutral": average_sentiment_probs[1],
                    "positive": average_sentiment_probs[2],
                }

                writer.writerow(avg_sentiments)

    def _predict_sentiment_probabilities(self, text) -> List:
        inputs = self._tokenizer(text, return_tensors="pt")
        outputs = self._model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probabilities.detach().numpy()[0]

    def _check_missing_data(self):
        data = pd.read_csv(
            self._config["data"][self._current_data_index]["raw"]
        )
        missing_values = data.isnull() | (data == "")
        missing_dates = []
        for index, row in missing_values.iterrows():
            if (
                row[RAW_ATTRIBUTES[0]]
                or row["created_utc"]
                or row[RAW_ATTRIBUTES[1]]
            ):
                date = data.at[index, RAW_ATTRIBUTES[0]]
                print(
                    f"Row {index + 1} has a missing or empty value. Date: {date}"
                )

        for i in range(len(data) - 1):
            date1 = datetime.strptime(data.at[i, RAW_ATTRIBUTES[0]], "%Y-%m-%d")
            date2 = datetime.strptime(
                data.at[i + 1, RAW_ATTRIBUTES[0]], "%Y-%m-%d"
            )

            days_diff = (date2 - date1).days
            if days_diff > 1:
                for j in range(1, days_diff):
                    next_date = date1 + timedelta(days=j)
                    if next_date.weekday() < 5:
                        missing_dates.append(next_date.strftime("%Y-%m-%d"))

        if missing_dates:
            print("Missing NYSE opening dates:")
            for missing_date in missing_dates:
                print(missing_date)
        else:
            print("No missing NYSE opening dates found.")

    @staticmethod
    def _check_and_adjust_unix_timestamp(
        unix_timestamp: int,
    ) -> Union[int, ValueError]:
        est = pytz.timezone("US/Eastern")
        dt = datetime.fromtimestamp(unix_timestamp, tz=pytz.utc).astimezone(est)
        if dt.hour == 9:
            return unix_timestamp
        elif dt.hour == 8:
            return unix_timestamp + 3600
        elif dt.hour == 10:
            return unix_timestamp - 3600
        else:
            raise ValueError("The provided timestamp is not close to 9 AM EST")

    def _obtain_data(self):
        create_file_if_not_exist(
            self._config["data"][self._current_data_index]["raw"]
        )
        self._comments = []
        nb_comments = 0
        end_date = self._end_date_unix
        start_date = end_date - 86400
        nb_retries = 0
        while nb_comments < 500 and nb_retries < 5:
            current_comments = list(
                self._push_shift.search_comments(
                    subreddit="wallstreetbets",
                    before=end_date,
                    after=start_date,
                    fields=[RAW_ATTRIBUTES[1], "created_utc"],
                    sort="asc",
                    limit=100,
                )
            )

            if current_comments:
                for comment in current_comments:
                    comment[RAW_ATTRIBUTES[1]] = comment.pop("body")
                self._comments.extend(current_comments)
                self._comments = self._keep_unique_values(self._comments)
                nb_comments = len(self._comments)
                end_date = self._comments[-1]["created_utc"]
                if end_date < (self._end_date_unix - 86400):
                    break
            else:
                time.sleep(300)
                nb_retries += 1

    def _filter_raw_data(self):
        self._comments = [
            {
                key: comment[key]
                if key != "permalink"
                else comment.get(key, "")
                for key in ["created_utc", RAW_ATTRIBUTES[1], "permalink"]
            }
            for comment in self._comments
            if "created_utc" in comment and RAW_ATTRIBUTES[1] in comment
        ]

        self._comments = [
            comment
            for comment in self._comments
            if comment[RAW_ATTRIBUTES[1]]
            not in ["[deleted]", "[removed]", "", None, "N/A"]
        ]

        comments_copy = deepcopy(self._comments)
        for item, comment in enumerate(comments_copy):
            self._comments[item]["date"] = self._current_date

        self._comments = sorted(self._comments, key=lambda x: x["date"])
        first_date = self._comments[0][RAW_ATTRIBUTES[0]]
        if not all(
            comment[RAW_ATTRIBUTES[0]] == first_date
            for comment in self._comments
        ):
            raise ValueError(f"Date are not all the same for current fetch")

    def _save_data_reddit(self):
        with open(
            self._config["data"][self._current_data_index]["raw"],
            "a",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    RAW_ATTRIBUTES[0],
                    "created_utc",
                    RAW_ATTRIBUTES[1],
                    "permalink",
                ],
            )
            if csvfile.tell() == 0:
                writer.writeheader()
            for comment in self._comments:
                writer.writerow(
                    {
                        "date": comment["date"],
                        "created_utc": comment["created_utc"],
                        RAW_ATTRIBUTES[1]: comment[RAW_ATTRIBUTES[1]],
                        "permalink": comment["permalink"],
                    }
                )


class MacroTrends(BaseDataProcessor):
    def __init__(
        self, specific_config, data_processor_helper, is_input_feature, **kwargs
    ):
        super().__init__(
            specific_config, data_processor_helper, is_input_feature, **kwargs
        )
    def _run_fetch(self):
        pass

    def _run_common_fetch(self):
        pass

class FRED(BaseDataProcessor):
    def __init__(
        self, specific_config, data_processor_helper, is_input_feature, **kwargs
    ):
        super().__init__(
            specific_config, data_processor_helper, is_input_feature, **kwargs
        )

    def _run_fetch(self) -> pd.DataFrame:
        request = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={self._config['data'][self._current_data_index].get('asset')}"
        request += f"&cosd={self._config['common']['start_date']}"
        request += f"&coed={self._config['common']['end_date']}"
        df = pd.read_csv(request, parse_dates=True)
        df.rename(
            columns={
                df.columns[0]: RAW_ATTRIBUTES[0],
                df.columns[1]: RAW_ATTRIBUTES[1],
            },
            inplace=True,
        )

        return df

class AlphaVantage(BaseDataProcessor):
    def __init__(
        self,
        specific_config,
        data_processor_helper,
        is_input_feature,
        **kwargs: Any,
    ):
        super().__init__(
            specific_config, data_processor_helper, is_input_feature, **kwargs
        )
        self._endpoints = read_json("resources/configs/av_api_support.json")
        self._av_key = os.getenv("ALPHA_VANTAGE_KEY")
        self._requests_last_minute = deque(maxlen=5)
        self._requests_today = 0
        self._rate_limiter = RateLimiter(max_calls=100, period=24*3600)

    def _fetch_data(self) -> Tuple[List, str]:
        if self._requests_today >= 100:
            logging.warning(
                f"Not fetching data for {self._config['data'][self._current_data_index]['asset']} "
                f"on Alpha Vantage. Reach the maximum of 100 on a day"
            )
            self._drop_current_asset()

        while len(self._requests_last_minute) == 5 and datetime.now() - self._requests_last_minute[0] < timedelta(minutes=1):
            time.sleep(5)

        with self._rate_limiter:
            data = self._config["data"][self._current_data_index].get("asset")
            endpoint = next(
                k
                for k, v in self._endpoints.items()
                if data in v["supported_symbols"]
            )
            input_keys = self._endpoints[endpoint]["response_helpers"][
                "input_key_name"
            ]
            concatenated_key = "".join(input_keys)
            self._endpoints[endpoint]["api_parameters"][concatenated_key] = data
            self._endpoints[endpoint]["api_parameters"]["apikey"] = self._av_key
            response = self._call_rest_api(
                self._endpoints[endpoint]["api_parameters"]
            ).json()

            response_data = response[
                self._endpoints[endpoint]["response_helpers"]["response"]
            ]


        self._requests_last_minute.append(datetime.now())
        self._requests_today += 1

        return response_data, endpoint

    def _run_fetch(self) -> pd.DataFrame:
        data_to_write, endpoint = self._fetch_data()
        return self._save_to_dataframe(
            endpoint,
            data_to_write
        )

    def _obtain_endpoint(self, key: str) -> Dict:
        for endpoint in self._endpoints:
            if key == endpoint:
                return endpoint

    def _call_rest_api(self, params: dict) -> requests.Response:
        url = "https://www.alphavantage.co/query"
        return requests.get(url=url, params=params)

    def _save_to_dataframe(
            self,
            endpoint: str,
            data: Union[Dict[str, Dict[str, str]], List[Dict[str, str]]]
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=self._endpoints[endpoint]["response_helpers"][
            "returned_values"])

        if self._endpoints[endpoint]["response_helpers"]["nested_response"]:
            df = self._write_nested_response(df, endpoint, data)
        else:
            df = self._write_non_nested_response(df, data)

        return df

    def _write_nested_response(
            self,
            df: pd.DataFrame,
            index: int,
            data: Dict[str, Dict[str, str]]
    ) -> pd.DataFrame:
        for date, point in reversed(data.items()):
            row = {
                col: point[col]
                for col in
                self._endpoints[index]["response_helpers"]["returned_values"][
                1:]
            }
            row[RAW_ATTRIBUTES[0]] = date
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df

    def _write_non_nested_response(
            self,
            df: pd.DataFrame,
            data: List[Dict[str, str]]
    ) -> pd.DataFrame:
        for item in reversed(data):
            df = pd.concat([df, pd.DataFrame([item])], ignore_index=True)
        return df