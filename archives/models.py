# ###############################################################################
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
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# RCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################


"""
Former models.py module that uses Darts
"""
import copy
from darts.timeseries import concatenate
from darts import TimeSeries
import numpy as np
from darts.metrics.metrics import rmse
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from typing import Union, Dict, Callable
import shutil
import datetime
import threading
import json
import pytz
from optuna.integration import PyTorchLightningPruningCallback
import torch

from sklearn.metrics import f1_score
from optuna import Trial
import optuna
from mean_reversion.config_utils import ConfigManager
from mean_reversion.utils import (
    save_json,
    read_json,
)
from mean_reversion.constants import MODEL_MAPPING, PICKLE, MODEL_OPTIMIZATION_PATH, MODELS_PATH
from pytorch_lightning.callbacks import ModelCheckpoint


class ModelArgsAssigner:
    def __init__(self,
                 config_manager : ConfigManager,
                 config : dict,
                 get_hyperparameters : Callable,
                 model_name : str):
        self._model_args = {}
        self._config_manager = config_manager
        self._config = config
        self._get_hyperparameters = get_hyperparameters
        self._model_name = model_name


    def _run_all(self):
        self._assign_hyperparameters()
        self._assign_fit_args()
        self._assign_predict_args()
        if self._is_predict_test:
            self._assign_test_args()
        self._assign_hyperparameters_with_keys()

    def _assign_hyperparameters(self) -> None:
        self._model_args['hyperparameters'] =\
            self._get_hyperparameters(self._model_name)

    def _assign_hyperparameters_with_keys(self) -> None:
        self._model_args['hyperparameters_with_keys'] = self._config_manager.get_model_args(self._model_name,'hyperparameters')

    @property
    def _is_predict_test(self) -> bool:
        return (len(self._config['common']['train_test_split']) >= 3 and self._config['common']['train_test_split'][2] != 0)


    def get_lr_scheduler_cls(self) -> Union[dict,None]:
        return self._config_manager.get_lr_scheduler_cls(model = self._model_name)

    def get_lr_scheduler_kwargs(self) -> Union[dict,None]:
        return self._config_manager.get_lr_scheduler_kwargs(model = self._model_name)

    def get_pl_trainer_kwargs(self) -> Union[dict,None]:
        return self._config_manager.get_pl_trainer_kwargs(model = self._model_name)

    def get_torch_metrics(self) -> Union[dict,None]:
        return self._config_manager.get_torch_metrics(model = self._model_name)

    def get_loss_fn(self) -> Union[dict,None]:
        return self._config_manager.get_loss_fn(model = self._model_name)

    def _assign_fit_args(self) -> None:
        fit_params = self._config_manager.get_model_args(self._model_name, 'fit',True)
        fit_args = {}
        if "num_loader_workers" in self._config["hyperparameters"]["common"]:
            fit_args["num_loader_workers"] = \
                self._config["hyperparameters"]["common"]["num_loader_workers"]

        for arg_name in fit_params:
            if arg_name in ["series", "val_series"]:
                df = pd.read_csv(
                    self._config["output"]["model_data"][
                        "train" if arg_name == "series" else "predict"
                    ]
                )

                fit_args[arg_name] = TimeSeries.from_dataframe(
                    df, "time", ["return"]
                ).astype(np.float32)
            elif self._config["inputs"]["future_covariates"]['data'] and arg_name in ["future_covariates", "val_future_covariates"]:
                df = pd.read_csv(
                    self._config["inputs"]["future_covariates"]["common"][
                        "model_data"
                    ]["train" if arg_name == "future_covariates" else "predict"]
                )
                fit_args[arg_name] = TimeSeries.from_dataframe(
                    df, "time", df.columns[df.columns != "time"].tolist()
                ).astype(np.float32)

            elif arg_name in ["past_covariates", "val_past_covariates"]:
                df = pd.read_csv(
                    self._config["inputs"]["past_covariates"]["common"][
                        "model_data"
                    ]["train" if arg_name == "past_covariates" else "predict"]
                )
                fit_args[arg_name] = TimeSeries.from_dataframe(
                    df, "time", df.columns[df.columns != "time"].tolist()
                ).astype(np.float32)
        self._model_args['fit'] = fit_args


    def _assign_predict_args(self):
        predict_args = {}
        model_predict_args = self._config_manager.get_model_args(self._model_name, 'predict', True)

        for arg_name in model_predict_args:
            if arg_name == "series":
                predict_args[arg_name] = self._model_args['fit'][arg_name]
            elif arg_name in ["past_covariates", "future_covariates"]:
                train_arg = (
                    "past_covariates"
                    if arg_name == "past_covariates"
                    else "future_covariates"
                )
                val_arg = (
                    "val_past_covariates"
                    if arg_name == "past_covariates"
                    else "val_future_covariates"
                )
                if (
                        train_arg in self._model_args['fit']
                        and val_arg in self._model_args['fit']
                        and self._model_args['fit'][train_arg] is not None
                        and self._model_args['fit'][val_arg] is not None
                ):
                    predict_args[arg_name] = concatenate(
                        [
                            self._model_args['fit'][train_arg],
                            self._model_args['fit'][val_arg],
                        ]
                    )
            elif arg_name == "n":
                predict_args[arg_name] = len(self._model_args['fit']["val_series"])
            elif arg_name in ["num_samples", "n_jobs"]:
                predict_args[arg_name] = self._config["hyperparameters"][
                    "common"
                ][arg_name]
        self._model_args['predict']= predict_args

    def _assign_test_args(self):
        test_args = {}
        model_predict_args = self._config_manager.get_model_args(self._model_name, 'predict', True)

        try :
            df_futures = pd.read_csv(
                self._config["inputs"]["future_covariates"]["common"][
                    "model_data"]["test"]
            )
            test_futures = TimeSeries.from_dataframe(
                df_futures, "time", df_futures.columns[df_futures.columns != "time"].tolist()
            ).astype(np.float32)


        except FileNotFoundError:
            pass

        df_past = pd.read_csv(
            self._config["inputs"]["past_covariates"]["common"][
                "model_data"]["test"]
        )
        test_past = TimeSeries.from_dataframe(
            df_past, "time", df_past.columns[df_past.columns != "time"].tolist()
        ).astype(np.float32)
        len_test = len(test_past)


        for arg_name in model_predict_args:
            if arg_name == "series":
                test_args[arg_name] = self._model_args['fit'][arg_name].append(
                    self._model_args['fit']["val_series"])
            elif arg_name in ["past_covariates", "future_covariates"]:
                train_arg = (
                    "past_covariates"
                    if arg_name == "past_covariates"
                    else "future_covariates"
                )
                val_arg = (
                    "val_past_covariates"
                    if arg_name == "past_covariates"
                    else "val_future_covariates"
                )
                if (
                        train_arg in self._model_args['fit']
                        and val_arg in self._model_args['fit']
                        and self._model_args['fit'][train_arg] is not None
                        and self._model_args['fit'][val_arg] is not None
                ):

                    test_args[arg_name] = concatenate(
                        [
                            self._model_args['fit'][train_arg],
                            self._model_args['fit'][val_arg],
                            test_past if train_arg == "past_covariates" else test_futures
                        ]
                    )
            elif arg_name == "n":
                test_args[arg_name] = len_test
            elif arg_name in ["num_samples", "n_jobs"]:
                test_args[arg_name] = self._config["hyperparameters"]["common"][arg_name]
        self._model_args['test']= test_args

    @property
    def model_args(self):
        return self._model_args

class HyperpametersOptimizer:
    def __init__(
        self,
        config_manager: ConfigManager,
    ):
        self._config_manager = config_manager
        self._config = config_manager.config
        self._model_args_assigner = \
            ModelArgsAssigner(config_manager, self._config, self._config_manager.get_model_hyperparameters_to_optimize, '')


    def _update_nested_dict(self, base_dict, updates_dict, target_dict):
        for k, v in base_dict.items():
            if k in updates_dict and isinstance(v, dict) and isinstance(
                    updates_dict[k], dict):
                target_dict[k] = self._update_nested_dict(v, updates_dict.get(k,
                                                                              {}),
                                                          target_dict.get(k,
                                                                          {}))
            else:
                target_dict[k] = updates_dict.get(k, v)
        return target_dict

    def run(self):
        n_trials = self._config['common']['hyperparameters_optimization'][
            'nb_trials']
        best_hyper_params = {}

        for model in self._config["hyperparameters_optimization"]["models"]:
            self._current_hyperparameters = {}
            best_hyper_params[model] = {}
            self._model_args_assigner._model_name = self._model = model
            self._model_args_assigner._hyperparameters_type = 'hyperparameters_optimization'
            self._model_args_assigner._run_all()
            self._model_args = self._model_args_assigner.model_args
            if self._config['common']['hyperparameters_optimization']['is_pruning']:

                pruner = optuna.pruners.MedianPruner(n_startup_trials=5,
                                                     n_warmup_steps=5,
                                                     interval_steps=5)
            else:
                pruner = None

            study = optuna.create_study(direction='minimize',pruner=pruner)
            study.optimize(self._objective, n_trials=n_trials,
                           show_progress_bar=True)

            best_hyper_params[model] = self._update_nested_dict(self._current_hyperparameters, study.best_params,
                               best_hyper_params[model])
            best_hyper_params[model]["work_dir"] = MODELS_PATH
            if 'lr_scheduler_kwargs' in best_hyper_params[model] and self._model_args_assigner.get_lr_scheduler_kwargs():
                best_hyper_params[model]['lr_scheduler_kwargs'] = self._model_args_assigner.get_lr_scheduler_kwargs()
            if 'lr_scheduler_cls' in best_hyper_params[model] and self._model_args_assigner.get_lr_scheduler_cls():
                best_hyper_params[model]['lr_scheduler_cls'] = self._model_args_assigner.get_lr_scheduler_cls()
            if 'pl_trainer_kwargs' in best_hyper_params[model] and self._model_args_assigner.get_pl_trainer_kwargs():
                best_hyper_params[model][
                    'pl_trainer_kwargs'] = self._model_args_assigner.get_pl_trainer_kwargs()
            if 'torch_metrics' in best_hyper_params[model] and self._model_args_assigner.get_torch_metrics():
                best_hyper_params[model][
                    'torch_metrics'] = self._model_args_assigner.get_torch_metrics()

            if 'loss_fn' in best_hyper_params[model] and self._model_args_assigner.get_loss_fn():
                best_hyper_params[model][
                    'loss_fn'] = self._model_args_assigner.get_loss_fn()


            with open(
                    f'{MODEL_OPTIMIZATION_PATH}/{model}/best_hyper_params.pkl',
                    'wb') as file:
                pickle.dump(best_hyper_params[model], file)

        self._config_manager.models_hyperparameters = best_hyper_params



    def _objective(self, trial: Trial):

        self._obtain_hyperparameters_to_optimize(trial)
        if 'callbacks' in self._current_hyperparameters[
            'pl_trainer_kwargs']:
            for callback in self._current_hyperparameters['pl_trainer_kwargs'][
                        'callbacks']:
                if isinstance(callback,
                              ModelCheckpoint) and callback.monitor == 'val_PortfolioReturnMetric':
                    callback.dirpath = os.path.join(MODEL_OPTIMIZATION_PATH,
                                                    self._current_hyperparameters[
                                                        'model_name'],
                                                    'portfolio_return_checkpoint')

        if self._config['common']['hyperparameters_optimization']['is_pruning']:
            pruning_callback = [
                PyTorchLightningPruningCallback(trial, monitor="val_loss")]
            if 'callbacks' in self._current_hyperparameters[
                'pl_trainer_kwargs']:
                self._current_hyperparameters['pl_trainer_kwargs'][
                    'callbacks'].extend(pruning_callback)
            else:
                self._current_hyperparameters['pl_trainer_kwargs'][
                    'callbacks'] = pruning_callback



        model = MODEL_MAPPING[self._model](
            **self._current_hyperparameters
        )

        model.fit(**self._model_args['fit'])

        if model.epochs_trained == 1:
            music_thread = threading.Thread(target=os.system('afplay super-mario-bros.mp3'))
            music_thread.start()
            raise ValueError(
                'Model only trained for one epoch. Training terminated prematurely.')

        predictions = model.predict(**self._model_args['predict']).median()
        actuals = self._model_args['fit']["val_series"]
        val_loss = rmse(
            pred_series=predictions, actual_series=actuals
        )
        if model.epochs_trained !=0:
            music_thread = threading.Thread(target=os.system('afplay super-mario-bros.mp3'))
            music_thread.start()
        return val_loss

    def _obtain_hyperparameters_to_optimize(self, trial: Trial) -> None:
        for hyperparameter, hyper_attributes in self._model_args['hyperparameters_with_keys'].items():
            self._process_hyperparameter(hyperparameter, hyper_attributes, trial)

    def _process_hyperparameter(self, hyperparameter, hyper_attributes, trial):
        suggest_methods = {
            'suggest_categorical': lambda hyper, trial, hyper_values: trial.suggest_categorical(
                hyper, hyper_values),
            'suggest_int': lambda hyper, trial, hyper_values: trial.suggest_int(
                hyper, min(hyper_values), max(hyper_values)),
            'suggest_float': lambda hyper, trial, hyper_values: trial.suggest_float(
                hyper, min(hyper_values), max(hyper_values)),
        }

        hyperparameter = tuple(hyperparameter) if isinstance(hyperparameter,
                                                             list) else (hyperparameter,)

        hyper_values = self._get_hyperparameter_value(hyperparameter)

        if hyperparameter[0] == "work_dir":
            self._current_hyperparameters["work_dir"] = MODEL_OPTIMIZATION_PATH
            return
        if not hyper_values:
            return
        if not hyper_attributes:
            self._set_hyperparameter_value(hyperparameter, self._get_hyperparameter_value(hyperparameter))
            return

        if isinstance(hyper_attributes,
                      dict) and 'trial_suggest' in hyper_attributes:
            trial_suggest = hyper_attributes['trial_suggest']
            if trial_suggest in suggest_methods:
                hyper_values = self._get_hyperparameter_value(hyperparameter)

                hyper_values = self._return_proper_hypervalues_format(
                    hyper_values)
                self._set_hyperparameter_value(hyperparameter,
                                               suggest_methods[trial_suggest](
                                                   hyperparameter[-1], trial,
                                                   hyper_values))

            else:
                raise ValueError(
                    f'{trial_suggest} is not in the accepted values for hyperparameter {hyperparameter}')
        elif isinstance(hyper_attributes, dict):
            for nested_hyperparameter, nested_values in hyper_attributes.items():
                self._process_hyperparameter(
                    list(hyperparameter) + [nested_hyperparameter],
                    nested_values, trial)

    def _get_hyperparameter_value(self, keys):
        d = copy.deepcopy(self._model_args['hyperparameters'])
        for key in keys:
            d = d.get(key)
            if d is None:
                return None
        return d

    def _set_hyperparameter_value(self, keys, value):
        d = self._current_hyperparameters
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def _return_proper_hypervalues_format(self, hyper_values: Union[int, float, list, tuple]) -> list:
        if not isinstance(hyper_values, (list, tuple)):
            hyper_values = [hyper_values, hyper_values]
        return hyper_values
    
class Modeler:
    def __init__(
        self,
        config_manager: ConfigManager,
    ):
        self._config_manager = config_manager
        self._config = self._config_manager.config
        self._model_args_assigner = \
            ModelArgsAssigner(config_manager, self._config, self._config_manager.get_model_hyperparameters, '')
        self._is_using_saved_model = (
            False if self._config["common"]["model_phase"] == "train" else True
        )

    def run(self):
        for model in self._config["hyperparameters"]["models"]:
            if model not in MODEL_MAPPING:
                raise ValueError(f"Invalid model: {model}")

            self._darts_model = [MODEL_MAPPING[model]]
            self._model_args_assigner._model_name = model
            self._model_args_assigner._run_all()

            self._model_args = self._model_args_assigner.model_args
            save_json(
                os.path.join(
                    self._model_args['hyperparameters']["work_dir"], model,
                    "run_information.json"
                ),
                {},
            )
            if self._config['common']['hyperparameters_optimization'][
                'is_optimizing'] or \
                    self._config['common']['hyperparameters_optimization']['is_using_optimized_hyper']:
                with open(
                        f'{MODEL_OPTIMIZATION_PATH}/{model}/best_hyper_params.pkl',
                        'rb') as file:
                    best_hyper_params = pickle.load(file)
            if self._config['common']['hyperparameters_optimization'][
                'is_optimizing']:
                self._compare_stored_hyperparameters_values(best_hyper_params)
            else :
                #À DÉVELOPPER = ASSIGNER LES VALEURS À self._model_args
                pass
            self._train_model(model)
            self._forecast_model(model)
            self._save_run_information(model)
        music_thread = threading.Thread(
            target=os.system('afplay super-mario-bros.mp3'))
        music_thread.start()
        print('Program finished successfully')


    def _forecast_model(self, model :str):
        self._is_new_model_better = True
        self._best_metrics = {}
        self._current_metrics = {}
        for is_evaluation_set in [True,False]:
            self._is_evaluation_set = is_evaluation_set
            if not self._is_evaluation_set and not self._model_args_assigner._is_predict_test:
                continue

            self._obtain_data_source()
            self._forecast_with_model()
            self._create_forecast_chart(model)
            self._coordinate_metrics(model)
            self._select_best_model(model)
        if self._is_new_model_better:
            self._save_best_model(model)

    def _obtain_data_source(self):
        if self._is_evaluation_set :
            self._data_for_predict = self._model_args['predict']
            self._actuals = self._model_args['fit']["val_series"]
            self._current_dataset = "evaluation"
        else :
            self._data_for_predict = self._model_args['test']
            df = pd.read_csv(
                self._config["output"]["model_data"]["test"])
            self._actuals = TimeSeries.from_dataframe(
                df, "time", ["return"]
            ).astype(np.float32)
            self._current_dataset = "test"



    def _is_builtin(self,obj) -> bool:
        if isinstance(obj, (int, float, str, bool, bytes, bytearray)):
            return True
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return all(self._is_builtin(item) for item in obj)
        elif isinstance(obj, dict):
            return all(self._is_builtin(k) and self._is_builtin(v) for k, v in obj.items())
        else:
            return False

    def _compare_dict_values(self,dict1, dict2) -> bool:
        common_keys = set(dict1.keys()) & set(dict2.keys())

        for key in common_keys:
            val1 = dict1[key]
            val2 = dict2[key]

            if self._is_builtin(val1) and self._is_builtin(val2) and val1 != val2:
                print(
                    f"Values are different for key '{key}': '{val1}' (dict1) vs '{val2}' (dict2)")
                return False

        return True


    def _save_run_information(self,model : str) -> None:
        with open(os.path.join(
                self._model_args['hyperparameters']["work_dir"], model,
                "run_information.json"
        ), 'r') as f:
            data = json.load(f)
        est = pytz.timezone('US/Eastern')
        now_in_est = datetime.datetime.now(est)
        date_str = now_in_est.strftime("%Y-%m-%d %H:%M")
        data['last_run_time'] = date_str
        save_json(
            os.path.join(
                self._model_args['hyperparameters']["work_dir"], model,
                "run_information.json"
            ),
            data,
        )

    def _compare_stored_hyperparameters_values(self,best_hyper_params):
        if 'is_optimizing' in self._config['common']['hyperparameters_optimization'] and \
            self._config['common']['hyperparameters_optimization']['is_optimizing']:
            if not self._compare_dict_values(best_hyper_params,
                                             self._model_args['hyperparameters']):
                music_thread = threading.Thread(
                    target=os.system('afplay super-mario-bros.mp3'))
                music_thread.start()
                raise ValueError(
                    f'Difference in optimized hyperparameters. Hyperparameters in '
                    f'pickle file and stored hyperparameters as variables'
                    f'are not the same')

    def _forecast_with_model(self) -> None:
        self._predictions = (
            self._darts_model[1].predict(**self._data_for_predict).median()
        )

    def _select_best_model(self, model: str) -> None:
        best_model_metrics_file = self._obtain_best_metrics_path()
        if best_model_metrics_file:
            self._best_metrics[self._current_dataset]= read_json(best_model_metrics_file)[self._current_dataset]
            self._current_metrics[self._current_dataset] = self._filter_relevant_metrics()
            if self._is_new_model_better:
                is_model_better_func = getattr(self, f"_is_model_better_{self._current_dataset}")
                self._is_new_model_better = is_model_better_func()
        else:
            self._save_best_model(model)

    def _save_best_model(self, model: str) -> None:
        model_dir = os.path.join(self._model_args['hyperparameters']["work_dir"], model)
        best_model_dir = os.path.join(
            self._config["common"]["best_model_path"], datetime.datetime.now().strftime("%Y%m%d"), model
        )
        best_root_dir = os.path.join(
            self._config["common"]["best_model_path"], datetime.datetime.now().strftime("%Y%m%d")
        )
        shutil.copytree(model_dir, best_model_dir)

        shutil.copy("config.yaml", best_root_dir)
        pickle_dir = os.path.join(best_root_dir, PICKLE)
        os.makedirs(pickle_dir, exist_ok=True)
        for file in self._config["inputs"]["past_covariates"]["common"][
            "pickle"
        ].values():
            shutil.copy(file, pickle_dir)

    def _obtain_best_metrics_path(self) -> Union[str, None]:
        most_recent_directory = self._obtain_most_recent_directory()
        if most_recent_directory:
            for root, _, files in os.walk(most_recent_directory):
                if "metrics.json" in files:
                    return os.path.join(root, "metrics.json")

        return None


    def _obtain_most_recent_directory(self) -> Union[str,None]:
        pattern = re.compile("^\d{8}$")
        directories = [
            d
            for d in os.listdir(self._config["common"]["best_model_path"])
            if os.path.isdir(os.path.join(self._config["common"]["best_model_path"], d))
        ]
        if directories :
            date_directories = [d for d in directories if pattern.match(d)]

            date_directories.sort(reverse=True)
            most_recent_directory = date_directories[0]

            return os.path.join(self._config["common"]["best_model_path"], most_recent_directory)
        return None

    def _is_model_better_evaluation(
        self) -> bool:
        for metric in self._current_metrics[self._current_dataset]:
            better_func = getattr(self, f"_is_{metric}_performance_better")
            if not better_func(self._current_metrics[self._current_dataset][metric], self._best_metrics[self._current_dataset][metric]):
                return False
        return True

    def _is_model_better_test(self) -> bool:
        if 'return_on_risk' in self._current_metrics[self._current_dataset]:
            if self._current_metrics['test']['return_on_risk'] < (.5 * self._current_metrics['evaluation']['return_on_risk']):
                return False
        return True

    def _is_return_performance_better(
        self, current: float, best: float
    ) -> bool:
        return current > best

    def _is_f1_score_performance_better(
        self, current: float, best: float
    ) -> bool:
        return current > best

    def _is_rmse_performance_better(self, current: float, best: float) -> bool:
        return current < best

    def _is_return_on_risk_performance_better(self, current: float, best: float) -> bool:
        return current > best

    def _is_predicted_vs_naive_performance_better(
        self, current: float, best: float
    ) -> bool:
        return current < best

    def _is_predicted_vs_naive_ma_performance_better(
        self, current: float, best: float
    ) -> bool:
        return current < best

    def _filter_relevant_metrics(self) -> Dict[str, float]:
        metrics_to_choose_model = self._config["common"][
            "metrics_to_choose_model"
        ]
        return {
            metric: self._metrics[metric] for metric in metrics_to_choose_model
        }

    def _train_model(self, model: str):
        if self._is_using_saved_model:
            if not 'torch_metrics' in self._model_args['hyperparameters'] or \
                    self._model_args['hyperparameters']['torch_metrics'] != 'PortfolioReturnMetric':

                self._darts_model.append(
                    self._darts_model[0].load_from_checkpoint(
                        model,
                        work_dir=self._model_args['hyperparameters']["work_dir"],
                        best=True,
                    )
                )
            else :
                checkpoint = torch.load(
                    f"{MODELS_PATH}/{model}/portfolio_return_checkpoint/best_model.ckpt")
                state_dict = checkpoint['state_dict']
                work_dir = os.path.join(self._model_args['hyperparameters']["work_dir"],model,'checkpoints')
                matching_files = [f for f in os.listdir(work_dir) if
                                  re.search(r'best-epoch=\d+', f)]

                if matching_files:
                    filename = matching_files[0]
                    match = re.search(r'best-epoch=(\d+)', filename)
                    last_epoch = int(match.group(1))
                else :
                    raise FileNotFoundError(f'File with best epoch for {model} does not exist')

                self._darts_model.append(
                    self._darts_model[0].load_from_checkpoint(
                        model,
                        work_dir=self._model_args['hyperparameters']["work_dir"],
                        best=True,
                    )
                )
                self._darts_model[1].n_epochs = last_epoch + 1
                self._darts_model[1].save_checkpoints = False
                self._darts_model[1].force_reset = False

                self._darts_model[1].fit(**self._model_args['fit'])
                cleaned_state_dict = {k.replace("model.", ""): v for k, v in
                                      state_dict.items()}
                self._darts_model[1].model.load_state_dict(cleaned_state_dict)

        else:
            self._darts_model.append(
                self._darts_model[0](**self._model_args['hyperparameters'])
            )

            if self._darts_model[1].epochs_trained != 0:
                music_thread = threading.Thread(
                    target=os.system('afplay super-mario-bros.mp3'))
                music_thread.start()
                raise ValueError(
                    f'epoch_trained should be 0, current value is {self._darts_model[1].epochs_trained} ')
            self._darts_model[1].fit(**self._model_args['fit'])

            if not self._darts_model[1].epochs_trained > 1:
                music_thread = threading.Thread(
                    target=os.system('afplay super-mario-bros.mp3'))
                music_thread.start()
                raise ValueError(
                    f'epoch_trained should be above 1, current value is {self._darts_model[1].epochs_trained} ')

            run_information = {}
            epochs_trained = self._darts_model[1].epochs_trained
            run_information['epochs_trained_after'] = epochs_trained

            save_json(
                os.path.join(
                    self._model_args['hyperparameters']["work_dir"], model,
                    "run_information.json"
                ),
                run_information,
            )

            monitor_value = ''
            for callback in self._model_args['hyperparameters']['pl_trainer_kwargs']['callbacks']:
                if isinstance(callback,
                              ModelCheckpoint):
                    monitor_value = callback.monitor
            if monitor_value == 'val_PortfolioReturnMetric':
                checkpoint = torch.load(
                    f"{MODELS_PATH}/{model}/portfolio_return_checkpoint/best_model.ckpt")
                state_dict = checkpoint['state_dict']
                self._model_args['hyperparameters']['n_epochs'] = 1

                self._model_args['hyperparameters']['save_checkpoints'] = False
                self._model_args['hyperparameters']['force_reset'] = False
                self._darts_model.append(
                    self._darts_model[0](**self._model_args['hyperparameters'])
                )
                self._darts_model[1].fit(**self._model_args['fit'])
                cleaned_state_dict = {k.replace("model.", ""): v for k, v in
                                      state_dict.items()}
                self._darts_model[1].model.load_state_dict(cleaned_state_dict)

            else :
                self._darts_model[1] = self._darts_model[0].load_from_checkpoint(
                    model,
                    work_dir=self._model_args['hyperparameters']["work_dir"],
                )

    def _create_forecast_chart(self, model: str) -> None:
        plt.figure(figsize=(8, 5))
        self._actuals.plot(label="actual")
        self._predictions.plot(label="forecast")
        plt.legend()

        plt.savefig(
            os.path.join(
                self._model_args['hyperparameters']["work_dir"], model, f"{self._current_dataset}_forecast.png"
            )
        )
        plt.close()

    def _coordinate_metrics(self, model: str) -> None:
        self._metrics = {}
        self._calculate_metrics_for_regression()
        self._calculate_metrics_for_classification()
        self._calculate_return()
        self._calculate_actual_return()
        try :
            json_data = read_json(os.path.join(self._model_args['hyperparameters']["work_dir"], model, "metrics.json"
                ))
            json_data[self._current_dataset] = self._metrics
        except FileNotFoundError:
            json_data = {}
            json_data[self._current_dataset] = self._metrics

        save_json(
            os.path.join(
                self._model_args['hyperparameters']["work_dir"], model, "metrics.json"
            ),
            json_data,
        )

    def _calculate_metrics_for_regression(self) -> None:
        self._metrics["rmse"] = rmse(
            pred_series=self._predictions, actual_series=self._actuals
        )


        ma_length = 20
        naive_forecast = self._actuals.shift(1)

        actuals_df = self._actuals.pd_dataframe()
        naive_forecast_ma_df = (
            actuals_df.rolling(window=ma_length).mean().shift(1)
        )
        naive_forecast_ma = TimeSeries.from_dataframe(naive_forecast_ma_df).astype(np.float32)

        self._metrics["predicted_vs_naive"] = self._metrics["rmse"] / rmse(
            pred_series=naive_forecast[1:], actual_series=self._actuals[1:]
        )
        self._metrics["predicted_vs_naive_ma"] = self._metrics["rmse"] / rmse(
            pred_series=naive_forecast_ma[ma_length:],
            actual_series=self._actuals[ma_length:],
        )

    def _calculate_metrics_for_classification(self):
        predictions_class = [1 if ret >= 0 else 0 for ret in
                                  self._predictions.values()]
        actuals_class = [1 if ret >= 0 else 0 for ret in
                               self._actuals.values()]
        self._metrics['f1_score'] = f1_score(actuals_class, predictions_class, average='weighted')


    def _calculate_return(self) -> None:
        portfolio_value = 1.0

        for pred, actual in zip(
            self._predictions.values(), self._actuals.values()
        ):
            if pred >= 0:
                portfolio_value *= 1 + actual
            else:
                portfolio_value *= 1 - actual


        cumulative_return = float(portfolio_value - 1.0)
        number_of_days = len(self._actuals)
        annualized_return = (1 + cumulative_return) ** (
                    252 / number_of_days) - 1

        daily_returns = [actual if pred >= 0 else - actual for
                         pred, actual in zip(self._predictions.values(),
                                             self._actuals.values())]
        daily_std_dev = np.std(daily_returns, ddof=1)
        annualized_risk = daily_std_dev * (
                    252 ** 0.5)

        return_on_risk = annualized_return / annualized_risk

        self._metrics["return"] = cumulative_return
        self._metrics["return_on_risk"] = return_on_risk


    def _calculate_actual_return(self) -> None:
        portfolio_value = 1.0
        for actual in self._actuals.values():
            portfolio_value *= 1 + actual

        cumulative_return = float(portfolio_value - 1.0)
        number_of_days = len(self._actuals)
        annualized_return = (1 + cumulative_return) ** (
                    252 / number_of_days) - 1

        daily_returns = [actual for actual in self._actuals.values()]
        daily_std_dev = np.std(daily_returns, ddof=1)
        annualized_risk = daily_std_dev * (
                    252 ** 0.5)

        return_on_risk = annualized_return / annualized_risk
        self._metrics["actual_return_on_risk"] = return_on_risk
        self._metrics["actual_return"] = float(portfolio_value - 1.0)
