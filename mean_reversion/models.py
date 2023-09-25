from pathlib import Path
import shutil
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from tensorboard.backend.event_processing import event_accumulator

from pytorch_forecasting import TimeSeriesDataSet, \
    RecurrentNetwork, DeepAR, TemporalFusionTransformer, NHiTS
from typing import Callable
from mean_reversion.utils import clear_directory_content, read_json
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import threading
from mean_reversion.config.model_config import  CUSTOM_MODEL
from mean_reversion.config.config_utils import ConfigManager

from pytorch_forecasting.metrics import MAE, SMAPE, \
    MultivariateNormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer


class Modeler:
    def __init__(
        self,
        config_manager: ConfigManager,
    ):
        self._config_manager = config_manager
        self._config = self._config_manager.config
        self._params = {}
        self._assign_params()

        self._datasets = ['train', 'predict', 'test']
        self._lightning_logs_dir = 'lightning_logs'

    def run(self):
        self._obtain_data()
        for model in self._config["hyperparameters"]["models"]:
            if model not in CUSTOM_MODEL:
                raise ValueError(f"Invalid model: {model}")

            self._model_name = model
            self._obtain_dataloader()
            self._model_dir = f'models/{self._model_name}'
            self._clean_directory()
            self._train_model()
            self._obtain_best_model()
            self._coordinate_predict()
            self._coordinate_interpretions()
            self._save_metrics_from_tensorboardflow()

        music_thread = threading.Thread(
            target=os.system('afplay super-mario-bros.mp3'))
        music_thread.start()
        print('Program finished successfully')

    def _clean_directory(self):
        clear_directory_content(self._model_dir)
        os.makedirs(self._model_dir, exist_ok=True)
        self._cleanup_logs(self._lightning_logs_dir)
        self._cleanup_logs(f'{self._lightning_logs_dir}/{self._model_name}')

    def _assign_params(self):
        model_config = read_json(
            "resources/configs/models_args.json"
        )["common"]
        for item in model_config.keys():
            if item in self._config["hyperparameters"]["common"]:
                self._params[item] = self._config["hyperparameters"]["common"][item]
            else :
                self._params[item] = None

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

    def _obtain_dataloader(self):


        for data_type in self._datasets:
            input_past = getattr(self, f'_input_past_{data_type}')
            input_future = getattr(self, f'_input_future_{data_type}',
                                   pd.DataFrame())
            output = getattr(self, f'_output_{data_type}')[['return']]

            if not input_future.empty:
                input_future = input_future.drop(columns=['time'])

            data = pd.concat([input_past, input_future, output], axis=1)
            data['group'] = 'group_1'
            setattr(self, f'_{data_type}_data', data)


        self._continuous_cols = [col for col in self._input_past_train.columns if col not in ["time"]]
        self._continuous_cols.append("return")
        add_encoder_length = True
        add_relative_time_idx = True
        add_target_scales = True
        static_categoricals = ["group"]

        if "RecurrentNetwork" in self._model_name or 'DeepAR' in self._model_name:
            self._continuous_cols = ["return"]
        if "NHiTS" in self._model_name :
            self._categorical_cols = []
            add_relative_time_idx = False
            add_encoder_length = False
            add_target_scales = False
            static_categoricals = []

        self._training_dataset = TimeSeriesDataSet(
            self._train_data,
            time_idx="time",
            target="return",
            group_ids=["group"],
            static_categoricals=static_categoricals,
            max_encoder_length=self._params["max_encoder_length"],
            max_prediction_length=self._params["max_prediction_length"],
            time_varying_known_categoricals=self._categorical_cols,
            time_varying_unknown_reals=self._continuous_cols,
            add_encoder_length=add_encoder_length,
            add_relative_time_idx= add_relative_time_idx,
            add_target_scales= add_target_scales,
            target_normalizer=GroupNormalizer(
                groups=["group"]
            ),

        )

        self._train_dataloader = self._training_dataset.to_dataloader(train=True, batch_size=self._params['batch_size'])
        self._predict_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, self._predict_data, predict=False, stop_randomization=True)
        self._predict_dataloader = self._predict_dataset.to_dataloader(train=False, batch_size=self._params['batch_size']*1000)
        self._test_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, self._test_data, predict=False, stop_randomization=True)
        self._test_dataloader = self._test_dataset.to_dataloader(train=False, batch_size=self._params['batch_size']*1000)

    def _train_model(self):
        pl.seed_everything(self._params['random_state'])
        model_to_train = self._config_manager.get_custom_model(self._model_name)

        self._model = model_to_train.from_dataset(
            dataset=self._training_dataset,
            **self._config_manager.models_hyperparameters[self._model_name],
        )


        callbacks_list = []
        if 'callbacks' in self._config['specific_config'][self._model_name] and self._config['specific_config'][self._model_name]['callbacks']:
           for callback in self._config['specific_config'][self._model_name]['callbacks']:
               if isinstance(callback,
                             ModelCheckpoint):
                    self._model_checkpoint = callback
                    callbacks_list.append(callback)
               if isinstance(callback,
                             EarlyStopping):
                   callbacks_list.append(callback)

        if not callbacks_list:
            callbacks_list = None

        self._logger = TensorBoardLogger(self._lightning_logs_dir, name= self._model_name)
        trainer = pl.Trainer(max_epochs=self._params["epochs"],
                             callbacks=callbacks_list,
                             logger=self._logger,
                             gradient_clip_val=self._params["gradient_clip_val"],
                             enable_model_summary=True,
                             )

        trainer.fit(self._model,
                    train_dataloaders=self._train_dataloader,
                    val_dataloaders=self._predict_dataloader)

    def _obtain_best_model(self):
        best_model_path = self._model_checkpoint.best_model_path
        self._best_model = self._model.load_from_checkpoint(best_model_path)
        #self._best_model = self._model.load_from_checkpoint('tempo/best_model.ckpt')


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
        self._save_multiple_interprations_plot(predictions_vs_actuals,
                                               features_sensitivity_path,
                                               self._best_model.plot_prediction_actual_by_variable)


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

            for j, ax in enumerate(
                    fig.get_axes()):
                title = ax.get_title()
                interpretation_file_path = os.path.join(directory_to_save,
                                                        f'{title}.png')


                fig.savefig(
                    interpretation_file_path)
                break

            plt.close(fig)

        plt.switch_backend(original_backend)


    def _calculate_metrics(self,dataloader, data):
        self._all_predicted_returns = []
        self._all_actual_returns = []
        self._time_indices = []
        self._cumulative_predicted_return = 1
        self._cumulative_actual_return = 1
        self._cumulative_max_possible = 1
        self._cumulative_index = len(data) - len(
            dataloader.dataset)

        predictions = self._best_model.predict(dataloader, mode ="raw", return_x=True, return_y = True)
        if len(predictions.output.prediction.shape) != 1:
            pass

        for i in range(predictions.output.prediction.shape[0]):
            current_prediction = predictions.output.prediction[i]

            predicted_return = current_prediction.median(dim=1).values

            actual_return = predictions.y[0][i].item()
            self._cumulative_predicted_return *= (
                        1 + actual_return) if predicted_return >= 0 else (
                        1 - actual_return)
            self._cumulative_actual_return *= (1 + actual_return)
            self._cumulative_max_possible *= (
                        1 + actual_return) if actual_return >= 0 else (
                        1 - actual_return)

            self._time_indices.append(data["time"].iloc[self._cumulative_index])
            self._cumulative_index += 1
            self._all_predicted_returns.append(predicted_return)
            self._all_actual_returns.append(actual_return)
        self._all_predicted_returns = [prediction.item() for prediction in self._all_predicted_returns]

    def _process_metrics(self,forecast_data):
        predicted_return_class = [1 if ret >= 0 else 0 for ret in
                                  self._all_predicted_returns]
        actual_return_class = [1 if ret >= 0 else 0 for ret in self._all_actual_returns]
        f1_score_value = f1_score(actual_return_class, predicted_return_class,
                                  average='weighted')
        rmse = np.sqrt(
            mean_squared_error(self._all_actual_returns, self._all_predicted_returns))
        naive_forecast = forecast_data["return"].rolling(20).mean()
        naive_forecast = naive_forecast[len(forecast_data) - len(self._all_actual_returns):].values

        naive_forecast_rmse = np.sqrt(
            mean_squared_error(self._all_actual_returns, naive_forecast))

        self._metrics = {
            "rmse": rmse,
            "f1_score": f1_score_value,
            "naive_forecast_rmse": naive_forecast_rmse,
            "rmse_vs_naive": rmse / naive_forecast_rmse,
            "return": self._cumulative_predicted_return - 1,
            "actual_return": self._cumulative_actual_return - 1,
            "max_possible_return": self._cumulative_max_possible - 1,
        }


    def _save_metrics(self,data_type):
        metrics_path = os.path.join(
            f'{self._model_dir}', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)
            existing_metrics[data_type] = self._metrics
        else:
            existing_metrics = {data_type: self._metrics}

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metrics, f, ensure_ascii=False, indent=4)


    def _plot_predictions(self,
                         data_type):
        plt.figure(figsize=(10, 6))
        plt.plot(self._time_indices, self._all_predicted_returns, color='blue',
                 label='Predicted Returns')
        plt.plot(self._time_indices, self._all_actual_returns, color='black',
                 label='Actual Returns')
        plt.xlabel('Time')
        plt.ylabel('Return')
        plt.title(f'Actual vs Predicted Returns over time - {data_type}')
        plt.legend()
        plt.savefig(os.path.join(f'models/{self._model_name}',
                                 f'actuals_vs_predictions_{data_type}.png'))


    def _coordinate_predict(self):

        for dataloader, data, data_type in [(self._predict_dataloader, self._predict_data, 'predict'), (self._test_dataloader, self._test_data, 'test')]:
            if dataloader is None or data is None:
                continue

            self._calculate_metrics( dataloader, data)
            self._process_metrics(data)
            self._save_metrics(data_type)
            self._plot_predictions(data_type)

    @staticmethod
    def _cleanup_logs(base_dir, keep=5):
        versions = [d for d in Path(base_dir).iterdir() if
                    d.is_dir() and 'version_' in d.name]
        sorted_versions = sorted(versions, key=os.path.getctime, reverse=True)
        for version in sorted_versions[keep:]:
            shutil.rmtree(version)

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