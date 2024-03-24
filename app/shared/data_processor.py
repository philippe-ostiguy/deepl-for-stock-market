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

import pandas as pd
import yfinance as yf
import numpy as np
from copy import deepcopy
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
torch.set_default_dtype(torch.float32)
from typing import Optional, List, Tuple, Union, Dict, Any, Callable
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from statsmodels.tsa.stattools import adfuller
import logging
from contextlib import suppress
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.decomposition import PCA
from datetime import datetime
load_dotenv()

from app.shared.config.config_utils import ConfigManager

from app.shared.utils import (
    obtain_market_dates,
    read_csv_to_pd_formatted,
    write_to_csv_formatted,
    write_pd_to_csv,
    get_previous_market_date,
    add_days_to_date
)
from app.shared.config.constants import (
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
        self._begin_train_index= []
        self._train_split_index = []
        self._predict_split_index = []
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

    @property
    def test_split_index(self):
        return self._test_split_index

    @test_split_index.setter
    def test_split_index(self, index):
        self._test_split_index = index

    def remove_first_transformed_data(
        self, data_to_process: pd.DataFrame
    ) -> pd.DataFrame:
        return data_to_process.iloc[ENGINEERED_DATA_TO_REMOVE:]

    def obtain_train_data(self, full_data_set: pd.DataFrame,
                          current_window: int) -> pd.DataFrame:
        return full_data_set.loc[
               self._begin_train_index[current_window]: self._train_split_index[
                   current_window]].copy()

    def obtain_predict_data(self, full_data_set: pd.DataFrame,
                            current_window: int) -> pd.DataFrame:
        return full_data_set.loc[self._train_split_index[current_window] + 1:
                                 self._predict_split_index[
                                     current_window]].copy()

    def obtain_test_data(self, full_data_set: pd.DataFrame) -> pd.DataFrame:
        return full_data_set.loc[self._test_split_index + 1:].copy()

    def _obtain_split_index_set(self) -> None:
        total_length = len(self._market_dates)
        self._test_split_index = int(total_length * (
                self._config["common"]["train_test_split"][0] +
                self._config["common"]["train_test_split"][1]))

        data_length_for_sliding = self._test_split_index
        train_ratio = self._config["common"]["train_test_split"][0]
        predict_ratio = self._config["common"]["train_test_split"][1]
        total_ratio = train_ratio + predict_ratio*self._config['common']['sliding_windows']

        for window in range(self._config['common']['sliding_windows']):
            train_index = int((train_ratio+window*predict_ratio)*data_length_for_sliding/total_ratio)
            begin_train_index = int(window*predict_ratio*data_length_for_sliding/total_ratio)
            self._train_split_index.append(train_index)
            self._begin_train_index.append(begin_train_index)


            if window == self._config['common']['sliding_windows'] - 1:
                predict_index = data_length_for_sliding
            else:
                predict_index = int((train_ratio + (
                            window + 1) * predict_ratio) * data_length_for_sliding / total_ratio)

            self._predict_split_index.append(predict_index)


    def write_ts_to_csv(
        self, window, path_write_file: Dict, columns_to_drop: Optional[List] = None
    ) -> None:
        self._convert_to_ts_data(columns_to_drop,window)
        self._write_ts_to_the_csv(path_write_file, window)

    def _write_ts_to_the_csv(self, path_to_write_file: Dict,window : int) -> None:
        write_to_csv_formatted(
            self._train_time_series, path_to_write_file["train"],sort_by_column_name='time',window = window
        )

        write_to_csv_formatted(
            self._predict_time_series, path_to_write_file["predict"], sort_by_column_name='time',window=window
        )
        write_to_csv_formatted(
            self._test_time_series, path_to_write_file["test"],sort_by_column_name='time'
        )

    def _convert_to_ts_data(self, columns_to_drop: Optional[List],window : int) -> None:
        if columns_to_drop:
            self._data_to_write_ts = self._data_to_write_ts.drop(
                columns=columns_to_drop
            )
        self._train_time_series = self.obtain_train_data(self._data_to_write_ts, window)
        self._train_time_series['time'] = self._train_time_series.index
        self._predict_time_series = self.obtain_predict_data(self._data_to_write_ts, window)
        self._predict_time_series['time'] = self._predict_time_series.index
        self._test_time_series = self.obtain_test_data(self._data_to_write_ts)
        self._test_time_series['time'] = self._test_time_series.index



class DataForModelSelector:
    def __init__(self, config_manager: ConfigManager):
        self._config = config_manager.config
        self._data_for_model_train = None
        self._data_for_model_predict = None
        self._data_for_model_test = None
        self._datasets = DATASETS

    def run(self) -> None:
        for window in range(self._config['common']['sliding_windows']):
            last_data = {}
            for dataset in self._datasets:
                for sources in self._config["output"]:
                    for data in sources['data']:
                        file = data["engineered"][dataset]
                        if not os.path.exists(file):
                            pass

                        if not hasattr(self, '_all_output'):
                            self._all_output = read_csv_to_pd_formatted(file, sort_by_column_name='time',
                                                                        window = window if dataset.lower() != 'test' else '')
                            last_data = data
                            continue
                        new_data = read_csv_to_pd_formatted(file, 'time',
                                                            window=window if dataset.lower() != 'test' else '')

                        if not (new_data['date'].equals(self._all_output['date']) and new_data['time'].equals(self._all_output['time'])):
                            raise ValueError(
                                f"Mismatch in 'time' and 'date' values for files in loop item {window}")
                        if 'date' in new_data.columns:
                            new_data.drop('date', axis = 1, inplace=True)
                        self._all_output= self._all_output.merge(new_data, on='time', how='outer')
                        last_data = data
                if not last_data:
                    raise ValueError('No output file found')
                write_to_csv_formatted(
                    self._all_output,
                    last_data["model_data"][dataset],
                    'time', window=window if dataset.lower() != 'test' else '')
                del self._all_output


            self._output_data_train = read_csv_to_pd_formatted(
                last_data["model_data"]['train'], "time", window=window
            )

            for current_source in self._config["inputs"]["past_covariates"][
                "sources"
            ]:
                for current_input in current_source["data"]:
                    train_file_path = current_input["engineered"]["train"].replace('.csv', f'_{window}.csv')
                    predict_file_path = current_input["engineered"]["predict"].replace('.csv', f'_{window}.csv')
                    test_file_path = current_input["engineered"]["test"]
                    if not os.path.exists(test_file_path):
                        continue
                    asset_name = current_input["asset"]
                    self._tempo_data_train = self._load_and_prefix_data(
                        train_file_path, asset_name, ["time","date"]
                    )

                    self._tempo_data_predict = self._load_and_prefix_data(
                        predict_file_path, asset_name, ["time","date"]
                    )

                    self._tempo_data_test = self._load_and_prefix_data(
                        test_file_path, asset_name, ["time","date"]
                    )

                    if not self._output_data_train["time"].equals(
                        self._tempo_data_train["time"]
                    ):
                        raise ValueError(
                            f"time values are not consistent in {asset_name}"
                        )

                    self._update_data_models(asset_name=asset_name)

            if self._config["common"]["model_phase"] == "train" and self._config["common"]["features_engineering"]["is_using_pca"]:
                self._make_features_removal()

            self._write_datasets_to_csv(window)
            self._data_for_model_train = None
            self._data_for_model_predict = None
            self._data_for_model_test = None

        self._check_data_before_model()


    def _update_data_models(self,asset_name):
        if self._data_for_model_train is None:
            self._set_initial_data()
        else:
            self._merge_with_existing_data(asset_name)


    def _set_initial_data(self):
        for dataset_type in DATASETS:
            setattr(self, f'_data_for_model_{dataset_type}', getattr(self,f'_tempo_data_{dataset_type}'))


    def _merge_with_existing_data(self, asset_name):
        duplicated_columns = self._get_duplicated_columns(self._tempo_data_train)
        if duplicated_columns:
            raise ValueError(f"Duplicate columns found for asset {asset_name}: {', '.join(duplicated_columns)}")

        self._data_for_model_train = self._merge_with_check(self._data_for_model_train, self._tempo_data_train, ['time', 'date'])
        self._data_for_model_predict = self._merge_with_check(self._data_for_model_predict, self._tempo_data_predict, ['time', 'date'])
        self._data_for_model_test = self._merge_with_check(self._data_for_model_test, self._tempo_data_test, ['time', 'date'])

    def _merge_with_check(self, df1, df2, keys):
        for key in keys:
            if not df1[key].equals(df2[key]):
                raise ValueError(f"Values in the '{key}' column do not match between dataframes.")
        df2 = df2.drop(columns=['date'])

        return pd.merge(df1, df2, on="time", how="left")

    def _get_duplicated_columns(self, data_train):
        duplicated_columns = self._data_for_model_train.columns.intersection(data_train.columns)
        return [col for col in duplicated_columns if col not in ["time", "date"]]

    def _write_datasets_to_csv(self, window : int):    
        for dataset_type in DATASETS:
            dataset = getattr(self, f'_data_for_model_{dataset_type}')
            file_path = self._config["inputs"]["past_covariates"]["common"]["model_data"][dataset_type]
    
            if dataset_type in ['train', 'predict']:
                write_to_csv_formatted(dataset, file_path, 'time', window=window)
            else:
                write_to_csv_formatted(dataset, file_path, 'time')

    
    def _check_data_before_model(self):
        folder_path = 'resources/input/model_data'
        all_files = os.listdir(folder_path)
        min_forecasts = self._config["common"]["min_validation_forecasts"]
        encoder_decoder_length =  self._config["hyperparameters"]["common"]["max_encoder_length"] + \
                                  self._config["hyperparameters"]["common"]["max_prediction_length"]
        for file in all_files :
            df = pd.read_csv(os.path.join(folder_path, file))
            if len(df) < (min_forecasts + encoder_decoder_length):
                raise ValueError(
                    f"Not enough data for {file}. Got {len(df)}, required at least {min_forecasts + encoder_decoder_length}"
                )

        for dataset in DATASETS:
            if dataset == 'test':
                test_files = [f for f in all_files if
                              f.endswith(f'test.csv')]
                dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file
                              in test_files]
                if not all(len(df) == len(dataframes[0])
                           and df[['time', 'date']].equals(dataframes[0][['time', 'date']]) for df in dataframes):
                    raise ValueError(
                        f"Mismatch in 'time' and 'date' values for test files")
                continue

            for window in range(self._config['common']["sliding_windows"]):
                loop_files = [f for f in all_files if
                              f.endswith(f'{dataset}_{window}.csv')]
                if len(loop_files) != 3:
                    raise ValueError(
                        f"Expected 3 files for loop item {window}, but found {len(loop_files)}")

                dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file
                              in loop_files]


                reference_date = dataframes[0]['date']
                reference_time = dataframes[0]['time']

                for df in dataframes[1:]:
                    if 'date' not in df or 'time' not in df:
                        raise ValueError(f'No "date or "time" for {df}')
                    if not (df['date'].equals(reference_date) and df['time'].equals(reference_time)):
                        raise ValueError(
                            f"Mismatch in 'time' and 'date' values for files in loop item {window}")

        last_time_in_predict = None

        for loop_item in range(self._config['common']["sliding_windows"]):

            train_files = [f for f in all_files if
                           f.endswith(f'train_{loop_item}.csv')]
            predict_files = [f for f in all_files if
                           f.endswith(f'predict_{loop_item}.csv')]

            train_dfs = [pd.read_csv(os.path.join(folder_path, file)) for file
                         in train_files]
            predict_dfs = [pd.read_csv(os.path.join(folder_path, file)) for file
                         in predict_files]
            for item, df in enumerate(train_dfs):
                if df['time'].iloc[-1] != predict_dfs[0]['time'].iloc[0] - 1:
                    raise ValueError(
                        f"Mismatch in 'time' values between train_{loop_item} and predict_{loop_item}")
            last_time_in_predict = predict_dfs[-1]['time'].iloc[-1]


        test_df = pd.read_csv(os.path.join(folder_path, 'input_past_test.csv'))

        if test_df['time'].iloc[0] != last_time_in_predict + 1:
            raise ValueError(
                "The first 'time' value in 'spy.csv' should be right after the last 'time' value in the last 'predict_{loop_item}.csv'")

    def _make_features_removal(self) -> None:
        self._train_time_data = self._data_for_model_train[['time','date']]
        self._test_time_data = self._data_for_model_test[['time','date']]
        self._predict_time_data = self._data_for_model_predict[['time','date']]
        self._data_for_model_train = self._data_for_model_train.drop(['time','date'],axis=1)
        self._data_for_model_test = self._data_for_model_test.drop(['time','date'], axis=1)
        self._data_for_model_predict = self._data_for_model_predict.drop(['time','date'], axis=1)

        pca = PCA()
        pca.fit(self._data_for_model_train)
        pca_variance = self._config["common"]["features_engineering"]["pca_variance"]

        n_components = sum(pca.explained_variance_ratio_.cumsum() <= pca_variance) + 1
        principal_components_train = pca.transform(self._data_for_model_train)[:,
                                    :n_components]
        principal_components_test = pca.transform(self._data_for_model_test)[:,
                                    :n_components]
        principal_components_predict = pca.transform(self._data_for_model_predict)[:,
                                    :n_components]

        self._data_for_model_train = pd.DataFrame(data=principal_components_train,
                                         columns=[f"PC{i+1}" for i in range(n_components)])

        self._data_for_model_test= pd.DataFrame(data=principal_components_test,
                                        columns=[f"PC{i+1}" for i in range(n_components)])
        self._data_for_model_predict= pd.DataFrame(data=principal_components_predict,
                                        columns=[f"PC{i+1}" for i in range(n_components)])


        self._data_for_model_train = self._data_for_model_train.join(self._train_time_data[['date', 'time']], how='left')
        self._data_for_model_predict = self._data_for_model_predict.join(self._predict_time_data[['date', 'time']], how='left')
        self._data_for_model_test = self._data_for_model_test.join(self._test_time_data[['date', 'time']], how='left')

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

    def _handle_missing_inf(self, data: pd.Series, check_missing_indices = True) -> Optional[pd.Series]:
        missing_or_inf = data.isna() | np.isinf(data)
        missing_or_inf_count = missing_or_inf.sum()
        missing_indices = set(range(1, len(data) + 1)) - set(data.index)
        if check_missing_indices :
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
        adjusted_data = adjusted_data.ffill()

        return adjusted_data

    def _make_identity(self, data: pd.Series, check_missing_indices = True) -> Optional[pd.Series]:
        data = self._handle_missing_inf(data, check_missing_indices = check_missing_indices)
        return data

    def _apply_identity(self, data: pd.Series) -> pd.Series:
        data = self._handle_missing_inf_apply(data)
        return data

    def _make_ratio(self, data: pd.Series, check_missing_indices = True) -> Optional[pd.Series]:
        ratio_data = (data / data.shift(1)) -1
        ratio_data = self._handle_missing_inf(ratio_data,check_missing_indices = check_missing_indices)
        return ratio_data

    def _apply_ratio(self, data: pd.Series) -> pd.Series:
        ratio_data = (data / data.shift(1))-1
        ratio_data = self._handle_missing_inf_apply(ratio_data)
        return ratio_data

    def _make_logarithmic(self, data: pd.Series, check_missing_indices = True) -> Optional[pd.Series]:
        data = np.log(data + 1e-7)
        data = self._handle_missing_inf(data,check_missing_indices = check_missing_indices)
        return data

    def _apply_logarithmic(self, data: pd.Series) -> pd.Series:
        data = np.log(data + 1e-7)
        data = self._handle_missing_inf_apply(data)
        return data

    def _make_first_difference(self, data: pd.Series, check_missing_indices = True) -> Optional[pd.Series]:
        data_diff = data.diff()
        data_diff = self._handle_missing_inf(data_diff,check_missing_indices = check_missing_indices)
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
                if self._processor._config["common"]["features_engineering"]["check_bell_shape"]:
                    if -2 <= skewness <= 7 and -2 <= kurtosis <= 7:
                        valid_transformations.append(
                            (name, p_value, skewness, kurtosis))

                else:
                    valid_transformations.append(
                        (name, p_value, skewness, kurtosis))
        self.TEMPO_FCT_MAKE_PLOT(data)
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

            self._data_processor_helper.write_ts_to_csv(self._processor._sliding_window,
                self._processor._config["data"][
                    self._processor._current_data_index
                ]["engineered"]

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
            self._engineered_data, self._processor._sliding_window
        ).copy()
        predict_data = self._data_processor_helper.obtain_predict_data(
            self._engineered_data, self._processor._sliding_window
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

        self._engineered_data = pd.concat([train_data, predict_data,test_data])

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
            self._engineered_data, self._processor._sliding_window
        )
        predict_data = self._data_processor_helper.obtain_predict_data(
            self._engineered_data, self._processor._sliding_window
        )
        test_data = self._data_processor_helper.obtain_test_data(
            self._engineered_data
        )
        first_index_test = test_data.index[0]
        first_index_train = train_data.index[0]
        check_missing_indices = False
        tempo_data = pd.DataFrame()
        tempo_test_data = pd.DataFrame()

        for column in self._columns_to_transform:
            self._current_column = column
            best_transformation = self.coordinate_stationarity(
                train_data[column].reset_index(drop=True)
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

                tempo_test_data[column] = dict(self._transformations)[best_transformation](test_data[column].reset_index(drop=True), check_missing_indices =check_missing_indices)
                tempo_data[column] = \
                    dict(self._transformations)[best_transformation]((pd.concat([train_data[column], predict_data[column]]).reset_index(drop=True)), check_missing_indices = check_missing_indices)


        tempo_data.index = range(first_index_train,first_index_train+len(tempo_data))
        tempo_test_data.index = range(first_index_test,first_index_test+len(tempo_test_data))
        tempo_test_data['date'] = test_data['date']
        tempo_data['date'] = pd.concat([train_data['date'], predict_data['date']])


        self._engineered_data = pd.concat([ tempo_data,tempo_test_data])
        if self._engineered_data.isna().any().any():
            raise ValueError("NA values found in the self._data_engineered")



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
        if 'open' in self._engineered_data.columns and 'close' in self._engineered_data.columns:
            self._engineered_data[target_column] = (
            self._engineered_data['close']
            / self._engineered_data["open"]
        ) - 1
        else :
            raise ValueError('Columns most have open and close name')

        self._engineered_data =\
            self._data_processor_helper.remove_first_transformed_data(
                self._engineered_data
            )
        columns_to_keep = [target_column,'date']
        self._engineered_data = self._engineered_data[columns_to_keep]

        train_data = self._data_processor_helper.obtain_train_data(
            self._engineered_data, self._processor._sliding_window
        ).copy()
        if self._processor._sliding_window !=0:
            train_data = self._data_processor_helper.remove_first_transformed_data(train_data)
        predict_data = self._data_processor_helper.obtain_predict_data(
            self._engineered_data,self._processor._sliding_window
        ).copy()

        test_data = self._data_processor_helper.obtain_test_data(
            self._engineered_data
        ).copy()

        self._data_processor_helper.data_to_write_ts = \
            pd.concat([train_data[columns_to_keep],
                       predict_data[columns_to_keep],
                       test_data[columns_to_keep]])
        self._data_processor_helper.write_ts_to_csv(self._processor._sliding_window,
            self._processor._config["data"][
                self._processor._current_data_index
            ]["engineered"]
        )


class BaseDataProcessor(ABC):
    def __init__(
        self,
        config : dict,
        config_manager : ConfigManager,
        data_processor_helper: DataProcessorHelper,
        is_input_feature: Optional[bool] = True,
        model_phase: Optional[str] = MODEL_PHASES,
        **kwargs,
    ):
        super().__init__()
        self._running_app = config_manager.running_app
        self._config = config
        self._model_phase = model_phase
        self._is_current_asset_dropped = False
        self._sliding_window = 0
        self._attributes_to_delete = self._config['common']["attributes_to_discard"]
        if is_input_feature:
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
        last_market_date_raw,_ = get_previous_market_date(datetime.now())
        last_market_date = pd.Timestamp(last_market_date_raw)
        write_older_data = False
        write_newer_data = False
        if first_date > pd.to_datetime(self._config["common"]["start_date"]):
            write_older_data = True

        if last_date < pd.to_datetime(self._config["common"]["end_date"]) and last_date < last_market_date:
            write_newer_data = True

        if write_older_data and write_newer_data:
            self._raw_data = pd.concat([self._run_fetch(), self._raw_data],
                                       ignore_index=True)
        elif write_older_data:
            self._config["common"]["end_date"] = (first_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            self._raw_data = pd.concat([self._run_fetch(),self._raw_data],ignore_index=True)
        elif write_newer_data:
            self._config["common"]["start_date"] = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            self._raw_data = pd.concat([self._run_fetch(), self._raw_data],
                                       ignore_index=True)

        if write_older_data or write_newer_data:
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
            if self._running_app != 'trader':
                for window in range(self._config['common']["sliding_windows"]):
                    self._sliding_window = window
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


class FutureCovariatesProcessor:
    def __init__(self, config_manager, data_processor_helper, sliding_window : int):
        self._config = config_manager.config
        self._data_processor_helper = data_processor_helper
        self._sliding_window = sliding_window

    def _initialize_variables(self):
        market_dates = pd.DataFrame(
            {RAW_ATTRIBUTES[0]: self._data_processor_helper.market_dates.index})

        self._future_covariates = (
            self._data_processor_helper.remove_first_transformed_data(
                market_dates
            )
        )

        self._train_data = self._data_processor_helper.obtain_train_data(
            deepcopy(self._future_covariates), self._sliding_window
        )
        if self._sliding_window !=0:
            self._train_data = self._data_processor_helper.remove_first_transformed_data(self._train_data)
        self._predict_data = self._data_processor_helper.obtain_predict_data(
            deepcopy(self._future_covariates), self._sliding_window
        )
        self._test_data = self._data_processor_helper.obtain_test_data(
            deepcopy(self._future_covariates)
        )
        self._scaler = {}

    def run(self):
        self._initialize_variables()
        self._train_data = self._add_covariates(
            self._train_data.copy(), fit=True
        )
        self._train_data['date'] = self._train_data['date'].dt.strftime('%Y-%m-%d')

        self._predict_data = self._add_covariates(
            self._predict_data.copy(), fit=False
        )
        self._predict_data['date'] = self._predict_data['date'].dt.strftime('%Y-%m-%d')

        self._test_data = self._add_covariates(
            self._test_data.copy(), fit=False
        )
        self._test_data['date'] = self._test_data['date'].dt.strftime('%Y-%m-%d')

        self._data_processor_helper.data_to_write_ts = pd.concat(
            [self._train_data, self._predict_data, self._test_data]
        )
        self._data_processor_helper.write_ts_to_csv(self._sliding_window,
            self._config["inputs"]["future_covariates"]["common"]["model_data"]
        )

    def _add_covariates(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        covariates = self._config["inputs"]["future_covariates"]["data"]
        if covariates and "day" in covariates:
            df["day"] = df[RAW_ATTRIBUTES[0]].dt.day
            df["day"] = self._scale("day", df["day"], fit)
        if covariates and "month" in covariates:
            df["month"] = df[RAW_ATTRIBUTES[0]].dt.month
            df["month"] = self._scale("month", df["month"], fit)
        return df

    def _scale(self, feature: str, data: pd.Series, fit: bool) -> pd.Series:

        data_reshaped = data.values.reshape(-1, 1)
        if fit :
            self._scaler[feature] = MinMaxScaler()
            self._scaler[feature].fit(data_reshaped)
            return self._scaler[feature].transform(data_reshaped)
        else :
            return self._scaler[feature].transform(data_reshaped)

class FRED(BaseDataProcessor):
    def __init__(
        self, specific_config,config_manager, data_processor_helper, is_input_feature, **kwargs
    ):
        super().__init__(
            specific_config,config_manager, data_processor_helper, is_input_feature, **kwargs
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

class YahooFinance(BaseDataProcessor):
    def __init__(
        self, specific_config, config_manager, data_processor_helper, is_input_feature, **kwargs
    ):
        super().__init__(
            specific_config, config_manager,data_processor_helper, is_input_feature, **kwargs
        )

    def _run_fetch(self) -> pd.DataFrame:
        asset = self._config['data'][self._current_data_index].get('asset')
        new_date_str = add_days_to_date(self._config['common']['end_date'],1)

        data = yf.download(asset,
                           start=self._config['common']['start_date'],
                           end=new_date_str)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        data.columns = data.columns.str.lower()
        test_data = data[-120:]
        test_data = test_data.copy()
        test_data['return'] = test_data['close'] / test_data['open'] - 1
        std_dev = test_data['return'].std()
        if std_dev == 0:
            raise ValueError(f'std deviation is 0 for {asset} from yahoo finance data')

        return data

