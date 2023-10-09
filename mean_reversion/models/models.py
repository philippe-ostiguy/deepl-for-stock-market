from pathlib import Path
import shutil
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from typing import Callable, Union, Optional
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import threading
from mean_reversion.models.model_customizer import CustomNHiTS,CustomDeepAR,CustomTemporalFusionTransformer,CustomlRecurrentNetwork
from mean_reversion.config.config_utils import ConfigManager, ModelValueRetriver
from mean_reversion.utils import clear_directory_content, read_json, save_json
import pytz
import datetime
import re
import subprocess
import glob
import copy
import torch
import optuna
import pickle
from abc import ABC
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s',filemode='w')

CUSTOM_MODEL = {
    "NHiTS": CustomNHiTS,
    "DeepAR": CustomDeepAR,
    "TemporalFusionTransformer": CustomTemporalFusionTransformer,
    "RecurrentNetwork": CustomlRecurrentNetwork
}


class BaseModelBuilder(ABC):
    def __init__(
        self,
        config_manager: ConfigManager = ConfigManager(),
        values_retriver = ModelValueRetriver()
    ):
        self._config_manager = config_manager
        self._config = self._config_manager.config
        self._params = {}
        self._datasets = ['train', 'predict', 'test']
        self._model_dir = ''
        self._lightning_logs_dir = ''
        self._logger = None
        self._model_name = ''
        self._values_retriever = values_retriver


    def _assign_params(self, hyperparameters_phase : Optional[str] = 'hyperparameters'):
        params = {}
        model_config = read_json(
            "resources/configs/models_args.json"
        )["common"]
        for item in model_config.keys():
            if item in self._config[hyperparameters_phase]["common"]:
                params[item] = self._config[hyperparameters_phase]["common"][item]
            else :
                params[item] = None
        return params

    def _clean_directory(self, exclusions : Optional[list] = None):
        clear_directory_content(self._model_dir, exclusions)
        os.makedirs(self._model_dir, exist_ok=True)
        self._cleanup_logs(self._lightning_logs_dir)
        self._cleanup_logs(f'{self._lightning_logs_dir}/{self._model_name}')

    @staticmethod
    def _cleanup_logs(base_dir, keep=5):
        if not os.path.exists(base_dir):
            return
        versions = [d for d in Path(base_dir).iterdir() if
                    d.is_dir() and 'version_' in d.name]
        sorted_versions = sorted(versions, key=os.path.getctime, reverse=True)
        for version in sorted_versions[keep:]:
            shutil.rmtree(version)

    def _obtain_data(self):

        if self._config["inputs"]["future_covariates"]['data']:
            self._categorical_cols = \
                self._config["inputs"]["future_covariates"]['data']

            for dataset in self._datasets:
                input_future = pd.read_csv(
                    self._config["inputs"]["future_covariates"]["common"][
                        "model_data"][dataset])
                for category in self._categorical_cols :
                    input_future[category] = input_future[category].astype(str)
                setattr(self, f'_input_future_{dataset}', input_future)
        else:
            self._categorical_cols = []

        for dataset in self._datasets:
            input_past = pd.read_csv(
                f"resources/input/model_data/input_past_{dataset}.csv")
            input_past.columns = input_past.columns.str.replace('.', '_',
                                                                regex=True)
            setattr(self, f'_input_past_{dataset}', input_past)

            output = pd.read_csv(f"resources/input/model_data/output_{dataset}.csv")
            setattr(self, f'_output_{dataset}', output)

    def _assign_data_models(self):
        for dataset_type in self._datasets:
            input_past = getattr(self, f'_input_past_{dataset_type}')
            input_future = getattr(self, f'_input_future_{dataset_type}',
                                   pd.DataFrame())
            output = getattr(self, f'_output_{dataset_type}')[['target']]

            if not input_future.empty:
                input_future = input_future.drop(columns=['time'])

            data = pd.concat([input_past, input_future, output], axis=1)
            data['group'] = 'group_1'
            setattr(self, f'_{dataset_type}_data', data)

    def _obtain_dataloader(self):

        self._continuous_cols = [col for col in self._input_past_train.columns if col not in ["time"]]
        self._continuous_cols.append("target")
        self._add_encoder_length = True
        self._add_relative_time_idx = True
        self._add_target_scales = True
        self._static_categoricals = ["group"]

        if "RecurrentNetwork" in self._model_name or 'DeepAR' in self._model_name:
            self._continuous_cols = ["target"]
        if "NHiTS" in self._model_name :
            self._categorical_cols = []
            self._add_relative_time_idx = False
            self._add_encoder_length = False
            self._add_target_scales = False
            self._static_categoricals = []

        self._training_dataset = TimeSeriesDataSet(
            self._train_data,
            time_idx="time",
            target="target",
            group_ids=["group"],
            static_categoricals=self._static_categoricals,
            max_encoder_length=self._params["max_encoder_length"],
            max_prediction_length=self._params["max_prediction_length"],
            time_varying_known_categoricals=self._categorical_cols,
            time_varying_unknown_reals=self._continuous_cols,
            add_encoder_length=self._add_encoder_length,
            add_relative_time_idx= self._add_relative_time_idx,
            add_target_scales=self._add_target_scales,
            target_normalizer=GroupNormalizer(
                groups=["group"]
            )
        )


        self._train_dataloader = self._training_dataset.to_dataloader(train=True, batch_size=self._params['batch_size'])
        self._predict_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, self._predict_data, predict=False, stop_randomization=True)
        self._predict_dataloader = self._predict_dataset.to_dataloader(train=False, batch_size=self._params['batch_size']*5000)
        self._test_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, self._test_data, predict=False, stop_randomization=True)
        self._test_dataloader = self._test_dataset.to_dataloader(train=False, batch_size=self._params['batch_size']*5000)

    def _train_model(self, hyperparameters : dict, callback_phase: Optional[str] = 'callbacks'):
        pl.seed_everything(self._params['random_state'])
        model_to_train = CUSTOM_MODEL[self._model_name]

        self._model = model_to_train.from_dataset(
            dataset=self._training_dataset,
            **hyperparameters[self._model_name],
        )

        callbacks_list = []
        if getattr(self._config_manager, callback_phase, None):
            for callback in getattr(self._config_manager, callback_phase)[self._model_name]['callbacks']:
                if isinstance(callback,
                              ModelCheckpoint):
                    self._model_checkpoint = copy.deepcopy(callback)
                    callbacks_list.append(callback)
                if isinstance(callback,
                              EarlyStopping):
                    callbacks_list.append(copy.deepcopy(callback))

        if not callbacks_list:
            callbacks_list = None

        if self._lightning_logs_dir:
            self._logger = TensorBoardLogger(self._lightning_logs_dir,
                                             name=self._model_name)
        self._trainer = pl.Trainer(
            max_epochs=self._params["epochs"],
            callbacks=callbacks_list,
            logger=self._logger,
            gradient_clip_val=self._params["gradient_clip_val"],
            enable_model_summary=True,
        )

        self._trainer.fit(self._model,
                          train_dataloaders=self._train_dataloader,
                          val_dataloaders=self._predict_dataloader)


class ModelBuilder(BaseModelBuilder):
    def __init__(
        self,
        config_manager: ConfigManager,
    ):
        super().__init__()
        self._config_manager = config_manager
        self._config = self._config_manager.config
        self._params = self._assign_params()

        self._lightning_logs_dir = 'lightning_logs'
        self._lower_index, self._upper_index = self._values_retriever.confidence_indexes


    def run(self):
        self._obtain_data()
        for model in self._config["hyperparameters"]["models"]:

            self._model_name = model
            self._assign_data_models()
            self._obtain_dataloader()
            self._model_dir = f'models/{self._model_name}'
            self._clean_directory()
            self._train_model(self._config_manager.hyperparameters)
            self._obtain_best_model()
            self._coordinate_evaluation()
            self._save_metrics_from_tensorboardflow()
            self._coordinate_interpretions()
            self._save_run_information()
            self._coordinate_select_best_model()

        music_thread = threading.Thread(
            target=os.system('afplay super-mario-bros.mp3'))
        music_thread.start()
        print('Program finished successfully')



    def _obtain_best_model(self):
        best_model_path = self._model_checkpoint.best_model_path
        self._best_model = self._model.load_from_checkpoint(best_model_path)
        #self._best_model = self._model.load_from_checkpoint('tempo/TemporalFusionTransformer/best_model.ckpt')


    def _coordinate_interpretions(self):
        if "TemporalFusionTransformer" in self._model_name:
            self._interpret_features_importance()
        self._interpret_features_sensitivity()


    def _interpret_features_sensitivity(self):

        raw_predictions = self._best_model.predict(self._predict_dataloader,
                                           return_x=True)
        predictions_vs_actuals = self._best_model.calculate_prediction_actual_by_variable(
            raw_predictions.x, raw_predictions.output)
        features_sensitivity_path = f'{self._model_dir}/features_sensitivity'
        os.makedirs(features_sensitivity_path, exist_ok=True)
        try :
            self._save_multiple_interprations_plot(predictions_vs_actuals,
                                                   features_sensitivity_path,
                                                   self._best_model.plot_prediction_actual_by_variable)
        except :
            pass

    def _interpret_features_importance(self):
        raw_predictions = self._best_model.predict(self._predict_dataloader,
                                                   mode="raw",
                                                   return_x=True)

        interpretations = self._best_model.interpret_output(
            raw_predictions.output,
            reduction="sum"
        )

        features_importance_dir = f'{self._model_dir }/features_importance'
        os.makedirs(features_importance_dir, exist_ok=True)
        self._save_multiple_interprations_plot(interpretations,
                                               features_importance_dir,
                                               self._best_model.plot_interpretation)

    def _save_multiple_interprations_plot(self, interpretations,
                                          directory_to_save,
                                          plot_function: Callable) -> None:
        original_backend = plt.get_backend()
        plt.switch_backend("Agg")
        plot_function(interpretations)
        for i, fig_num in enumerate(plt.get_fignums()):
            fig = plt.figure(fig_num)

            for j, ax in enumerate(fig.get_axes()):
                title = ax.get_title()
                interpretation_file_path = os.path.join(directory_to_save,f'{title}.png')
                fig.savefig(interpretation_file_path)
                break

            plt.close(fig)

        plt.switch_backend(original_backend)


    def _coordinate_select_best_model(self):
        self._is_new_model_better = True
        self._best_metrics = {}
        self._current_metrics = {}
        for dataloader, data, dataset_type in [(self._predict_dataloader, self._predict_data, 'predict'), (self._test_dataloader, self._test_data, 'test')]:
            if dataloader is None or data is None:
                continue
            self._dataset_type = dataset_type
            self._select_best_model()
        if self._is_new_model_better:
            self._save_best_model()


    def _select_best_model(self) -> None:
        best_model_metrics_file = self._obtain_best_metrics_path()
        if best_model_metrics_file:
            self._best_metrics[self._dataset_type]= read_json(best_model_metrics_file)[self._dataset_type]
            self._current_metrics[self._dataset_type] = self._filter_relevant_metrics()
            if self._is_new_model_better:
                is_model_better_func = getattr(self, f"_is_model_better_{self._dataset_type}")
                self._is_new_model_better = is_model_better_func()
        else:
            self._save_best_model()

    def _obtain_best_metrics_path(self):
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

    def _save_best_model(self) -> None:

        best_model_dir = os.path.join(
            self._config["common"]["best_model_path"],
            datetime.datetime.now().strftime("%Y%m%d"),
            self._model_name
        )

        best_root_dir = os.path.join(
            self._config["common"]["best_model_path"],
            datetime.datetime.now().strftime("%Y%m%d")
        )
        if os.path.exists(best_root_dir):
            shutil.rmtree(best_root_dir)
        shutil.copytree(self._model_dir, best_model_dir)
        shutil.copy("config.yaml", best_root_dir)

        ckpt_files = glob.glob(os.path.join(best_model_dir, '*.ckpt'))
        for file in ckpt_files:
            os.remove(file)

        subprocess.run(['git', 'add',best_root_dir])


    def _is_model_better_predict(
        self) -> bool:
        for metric in self._current_metrics[self._dataset_type]:
            better_func = getattr(self, f"_is_{metric}_performance_better")
            if metric not in self._best_metrics[self._dataset_type]:
                return True
            if not better_func(self._current_metrics[self._dataset_type][metric], self._best_metrics[self._dataset_type][metric]):
                return False
        return True

    def _is_model_better_test(self) -> bool:
        if 'return_on_risk' in self._current_metrics[self._dataset_type]:
            if self._current_metrics['test']['return_on_risk'] < (.4 * self._current_metrics['predict']['return_on_risk']):
                return False
        return True

    def _is_return_performance_better(
        self, current: float, best: float
    ) -> bool:
        return current > best

    def _is_return_on_risk_performance_better(self, current: float, best: float) -> bool:
        return current > best


    def _filter_relevant_metrics(self) :
        metrics_to_choose_model = self._config["common"][
            "metrics_to_choose_model"
        ]
        return {
            metric: self._metrics[self._dataset_type][metric] for metric in metrics_to_choose_model
        }

    def _calculate_metrics(self,dataloader, data):
        self._all_predicted_returns = []
        self._all_actual_returns = []
        self._all_predicted_values = []
        self._all_actual_values = []
        self._preds_class = []
        self._actual_class = []
        self._time_indices = []
        self._cumulative_predicted_return = 1
        self._cumulative_actual_return = 1
        self._cumulative_index = len(data) - len(
            dataloader.dataset)
        self._nb_of_trades = 0
        max_drawdown = 0
        peak = self._cumulative_predicted_return

        for index in range(self._raw_predictions.output.prediction.shape[0]):
            current_prediction, indices = self._raw_predictions.output.prediction[index].sort()

            median_pred_value = current_prediction.median(dim=1).values
            lower_value = current_prediction[0,self._lower_index]
            upper_value = current_prediction[0,self._upper_index]

            actual_value = self._raw_predictions.y[0][index].item()

            if not self._config["common"]["make_data_stationary"] and index  == 0:
                continue

            if not self._config["common"]["make_data_stationary"]:
                actual_return = (actual_value/ self._raw_predictions.y[0][index -1].item()) -1
                median_pred_return = (median_pred_value / self._raw_predictions.y[0][index -1].item()) -1
                upper_return = (upper_value /  self._raw_predictions.y[0][index - 1].item())-1
                lower_return = (lower_value / self._raw_predictions.y[0][index - 1].item())-1

            else :
                median_pred_return = median_pred_value
                lower_return = lower_value
                upper_return = upper_value
                actual_return = actual_value

            if lower_return > 0 and upper_return>0:
                self._cumulative_predicted_return *= (
                        1 + actual_return)
                self._nb_of_trades +=1
                self._preds_class.append(median_pred_return)
                self._actual_class.append(actual_return)

            elif upper_return < 0 and lower_return < 0:
                self._cumulative_predicted_return *= (1 - actual_return)
                self._nb_of_trades += 1
                self._preds_class.append(median_pred_return)
                self._actual_class.append(actual_return)

            if self._cumulative_predicted_return > peak:
                peak =  self._cumulative_predicted_return
            else:
                drawdown = (peak - self._cumulative_predicted_return) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown


            self._cumulative_actual_return *= (1 + actual_return)

            self._time_indices.append(data["time"].iloc[self._cumulative_index])
            self._cumulative_index += 1
            self._all_predicted_returns.append(median_pred_return)
            self._all_actual_returns.append(actual_return)
            self._all_predicted_values.append(median_pred_value)
            self._all_actual_values.append(actual_value)

        self._all_predicted_returns = [prediction.item() for prediction in self._all_predicted_returns]
        self._all_predicted_values = [prediction.item() for prediction in
                                       self._all_predicted_values]
        self._max_drawdown = max_drawdown

    def _process_metrics(self, dataset_type, forecast_data):
        predicted_return_class = [1 if ret >= 0 else 0 for ret in
                                  self._preds_class]
        actual_return_class = [1 if ret >= 0 else 0 for ret in self._actual_class]
        f1_score_value = f1_score(actual_return_class, predicted_return_class,
                                  average='weighted')
        rmse = np.sqrt(
            mean_squared_error(self._all_actual_returns, self._all_predicted_returns))
        if self._params["max_encoder_length"] <= 20:
            rolling_windows = self._params["max_encoder_length"] - 1
        else :
            rolling_windows = 20
        naive_forecast = forecast_data["target"].rolling(rolling_windows).mean()
        naive_forecast = naive_forecast[len(forecast_data) - len(self._all_actual_returns):].values

        naive_forecast_rmse = np.sqrt(
            mean_squared_error(self._all_actual_returns, naive_forecast))

        num_days = len(self._all_actual_returns)
        annualized_return = (self._cumulative_predicted_return ) ** (252 / num_days) - 1
        actual_annualized_return = (self._cumulative_actual_return)** (252 / num_days) - 1
        actual_daily_returns = np.array(
            self._all_actual_returns)
        annualized_risk = np.std(actual_daily_returns) * (252 ** 0.5)
        return_on_risk = annualized_return / annualized_risk if annualized_risk != 0 else 0
        actual_return_on_risk = actual_annualized_return / annualized_risk if annualized_risk != 0 else 0
        if not hasattr(self, '_metrics'):
            self._metrics = {}
        self._metrics[dataset_type] = {
            "rmse": rmse,
            "f1_score": f1_score_value,
            "naive_forecast_rmse": naive_forecast_rmse,
            "rmse_vs_naive": rmse / naive_forecast_rmse,
            "return": self._cumulative_predicted_return - 1,
            "actual_return": self._cumulative_actual_return - 1,
            "return_on_risk" : return_on_risk,
            "actual_return_on_risk" : actual_return_on_risk,
            "max_drawdown" : self._max_drawdown,
            "nb_of_trades" : self._nb_of_trades
        }


    def _save_metrics(self,dataset_type):
        metrics_path = os.path.join(
            f'{self._model_dir}', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)
            existing_metrics[dataset_type] = self._metrics[dataset_type]
        else:
            existing_metrics = {dataset_type: self._metrics[dataset_type]}

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metrics, f, ensure_ascii=False, indent=4)


    def _plot_predictions(self, dataset_type):

        plt.figure(figsize=(10, 6))
        plt.plot(self._time_indices, self._all_predicted_values, color='blue',
                 label='Predicted Values')
        plt.plot(self._time_indices, self._all_actual_values, color='black',
                 label='Actual Values')
        plt.xlabel('Time')
        plt.ylabel('Output')
        plt.title(f'Actual vs Predicted Values over time - {dataset_type}')
        plt.legend()
        plt.savefig(os.path.join(f'models/{self._model_name}',
                                 f'actuals_vs_predictions_{dataset_type}.png'))
        plt.close()


    def _coordinate_evaluation(self):

        for dataloader, data, dataset_type in [(self._predict_dataloader, self._predict_data, 'predict'), (self._test_dataloader, self._test_data, 'test')]:
            if dataloader is None or data is None:
                continue
            self._raw_predictions = self._best_model.predict(dataloader,
                                                             mode="raw",
                                                             return_x=True,
                                                             return_y=True)
            self._calculate_metrics(dataloader, data)
            self._process_metrics(dataset_type,data)
            self._save_metrics(dataset_type)
            self._plot_predictions(dataset_type)

    def _save_metrics_from_tensorboardflow(self):
        metrics_dict = {}

        os.makedirs(f'{self._model_dir}/tensorboard', exist_ok=True)
        for event_file in os.listdir(self._logger.log_dir):
            if not event_file.startswith('events.out.tfevents'):
                continue
            full_path = os.path.join(self._logger.log_dir, event_file)
            ea = event_accumulator.EventAccumulator(full_path)
            ea.Reload()

            for tag in ea.Tags()['scalars']:
                metrics_dict[tag] = ea.Scalars(tag)

        for metric, scalars in metrics_dict.items():
            plt.figure(figsize=(10, 5))

            if metric == 'train_loss_step':
                steps = [scalar.step for scalar in scalars]
            else:
                steps = list(range(len(scalars)))

            values = [scalar.value for scalar in scalars]
            plt.plot(steps, values, label=metric)
            plt.xlabel('Steps' if metric == 'train_loss_step' else 'Epoch')
            plt.ylabel('Value')
            plt.title(metric)
            plt.legend(loc='upper right')
            plt.savefig(f"{self._model_dir}/tensorboard/{metric.replace('/', '_')}.png")
            plt.close()


    def _save_run_information(self) -> None:
        data = {}
        est = pytz.timezone('US/Eastern')
        now_in_est = datetime.datetime.now(est)
        date_str = now_in_est.strftime("%Y-%m-%d %H:%M")
        data['last_run_time'] = date_str
        data['last_epoch_trained'] = self._trainer.current_epoch + 1
        save_json(
            os.path.join(
                self._model_dir,
                "run_information.json"
            ),
            data,
        )


class HyperpametersOptimizer(BaseModelBuilder):
    def __init__(
            self,
            config_manager: ConfigManager

    ):
        super().__init__()
        self._config_manager = config_manager
        self._config = config_manager.config
        #self._lightning_logs_dir = 'lightning_logs/model_optimization'
        self._params_to_optimized = self._assign_params(hyperparameters_phase='hyperparameters_optimization')

        self._model_suggested_type = {}

    def run(self):
        n_trials = self._config['common']['hyperparameters_optimization'][
            'nb_trials']

        self._obtain_data()
        self._current_hyperparameters = {}
        for model in self._config["hyperparameters_optimization"]["models"]:
            self._current_hyperparameters['model'] = {}
            self._model_suggested_type = \
                self._config_manager.get_model_suggest_type(model)
            self._model_name = model
            self._assign_data_models()
            optuna_storage = 'last_study.db'
            self._model_dir = f'models/hyperparameters_optimization/{self._model_name}'
            self._clean_directory(exclusions=[optuna_storage])
            if self._config['common']['hyperparameters_optimization'][
                'is_pruning']:

                pruner = optuna.pruners.MedianPruner(n_startup_trials=5,
                                                     n_warmup_steps=5,
                                                     interval_steps=5)
            else:
                pruner = None
            sampler = optuna.samplers.TPESampler(seed=42)
            storage_name = f"sqlite:///{os.path.join(self._model_dir, optuna_storage)}"
            if os.path.exists(os.path.join(self._model_dir, optuna_storage)):
                  os.remove(os.path.join(self._model_dir, optuna_storage))


            try :
                study = optuna.load_study(pruner = pruner,
                                            study_name= optuna_storage.replace('.db',''),
                                            storage= storage_name,
                                            sampler=sampler)
            except KeyError as key_error:
                logging.warning(f"Study name doesn't exists: {key_error}")

                study = optuna.create_study(direction='maximize',
                                            pruner = pruner,
                                            study_name= optuna_storage.replace('.db',''),
                                            storage= storage_name,
                                            sampler=sampler,
                                            load_if_exists=True)

            study.optimize(self._objective, n_trials=n_trials,show_progress_bar=True)

            with open(f"{self._model_dir}/best_study.pkl", "wb") as fout:
                pickle.dump(study, fout)
            print(study.best_params)
            self._values_retriever.confidence_indexes = ''


    def _objective(self, trial: optuna.Trial):
        self._hyper_possible_values = self._config_manager.hyperparameters_to_optimize[self._model_name]
        self._current_suggested_type = self._model_suggested_type
        self._current_hyperparameters[self._model_name] = self._assign_hyperparameters(trial)

        self._hyper_possible_values = self._params_to_optimized
        self._current_suggested_type = {**self._model_suggested_type,**self._config_manager.get_model_suggest_type('common', model_argument_type='')}
        self._params = self._assign_hyperparameters(trial)

        self._adjust_hyperparameters()
        self._obtain_dataloader()
        self._train_model(self._current_hyperparameters,callback_phase='callbacks_for_optimization')

        if self._model.current_epoch == 0 or self._model.current_epoch ==1:
            music_thread = threading.Thread(
                target=os.system('afplay super-mario-bros.mp3'))
            music_thread.start()
            raise ValueError(
                'Model only trained for one epoch. Training terminated prematurely.')
        checkpoint = torch.load(
            f"{self._model_dir}/best_model.ckpt")

        model_checkpoint_key = next(
            key for key in checkpoint["callbacks"] if "ModelCheckpoint" in key)

        best_value = checkpoint["callbacks"][model_checkpoint_key][
            'best_model_score'].item()
        print(f'current best return : {best_value}')

        return best_value

    @staticmethod
    def _find_closest_value(lst, K, exclude):
        return min((abs(val - K), val) for val in lst if val not in exclude)[1]


    def _adjust_hyperparameters(self):
        if 'likelihood' in self._params and 'confidence_level' in self._params and\
                self._params['confidence_level'] != 0.5 and self._params['confidence_level'] \
                not in self._params['likelihood'] and (1-self._params['confidence_level']) \
                not in self._params['likelihood']:
            to_remove_1 = self._find_closest_value(self._params['likelihood'], self._params['confidence_level'],
                                  exclude=[0.5])
            self._params['likelihood'].remove(to_remove_1)

            to_remove_2 = self._find_closest_value(self._params['likelihood'], 1 - self._params['confidence_level'],
                                  exclude=[0.5])
            self._params['likelihood'].remove(to_remove_2)
            self._params['likelihood'].append(self._params['confidence_level'])
            self._params['likelihood'].append(1 - self._params['confidence_level'])

        self._params['likelihood'].sort()

        self._current_hyperparameters[self._model_name]['loss'] = ConfigManager.assign_loss_fct(self._current_hyperparameters[self._model_name],self._params)['loss']
        index_high_quantile = self._params['likelihood'].index(self._params['confidence_level'])
        index_low_quantile = self._params['likelihood'].index(1 - self._params['confidence_level'])
        self._values_retriever.confidence_indexes = (index_low_quantile,index_high_quantile)

    def _assign_hyperparameters(self, trial: optuna.Trial) -> dict:
        hyperparameters_value = {}
        for hyperparameter, hyper_properties in self._current_suggested_type.items():
            current_value = self._process_hyperparameter(hyperparameter, hyper_properties,
                                         trial)
            if current_value:
                hyperparameters_value[hyperparameter] = current_value
        return hyperparameters_value

    def _process_hyperparameter(self, hyperparameter, hyper_properties, trial):
        suggest_methods = {
            'suggest_categorical': lambda hyper, trial,
                                          hyper_values: trial.suggest_categorical(
                hyper, hyper_values),
            'suggest_int': lambda hyper, trial, hyper_values: trial.suggest_int(
                hyper, min(hyper_values), max(hyper_values)),
            'suggest_float': lambda hyper, trial,
                                    hyper_values: trial.suggest_float(
                hyper, min(hyper_values), max(hyper_values)),
        }


        hyper_values = self._get_hyperparameter_value(hyperparameter)

        if not hyper_values:
            return None

        if not isinstance(hyper_values, list):
            return hyper_values

        if not 'trial_suggest' in hyper_properties:
            return hyper_values

        trial_suggest = hyper_properties['trial_suggest']
        return suggest_methods[trial_suggest](hyperparameter, trial,hyper_values)


    def _get_hyperparameter_value(self, hyperparameter):
        hyperparameters_dict = copy.deepcopy(self._hyper_possible_values)
        if hyperparameter in hyperparameters_dict:
            return hyperparameters_dict[hyperparameter]
        return None