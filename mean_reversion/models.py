
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, RecurrentNetwork
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

class Modeler:
    def __init__(
        self,
        config_manager: ConfigManager,
    ):
        self._config_manager = config_manager
        self._config = self._config_manager.config
        self._params = {}
        self._assign_params()


    def run(self):
        self._obtain_data()
        #clear_directory_content('models/tempo')
        for model in self._config["hyperparameters"]["models"]:
            if model not in CUSTOM_MODEL:
                raise ValueError(f"Invalid model: {model}")

            self._model_name = model
            self._obtain_dataloader()
            source_dir = f'models/{self._model_name}'
            clear_directory_content(source_dir)
            os.makedirs(f'{source_dir}/{self._model_name}')
            self._train_model()
            self._coordinate_predict()


        music_thread = threading.Thread(
            target=os.system('afplay super-mario-bros.mp3'))
        music_thread.start()
        print('Program finished successfully')

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

            self._categorical_cols = self._config["inputs"]["future_covariates"]['data']
            self._input_future_predict = pd.read_csv(self._config["inputs"]["future_covariates"]["common"]["model_data"]["predict"])
            self._input_future_train = pd.read_csv(self._config["inputs"]["future_covariates"]["common"]["model_data"]["train"])
            self._input_future_test = pd.read_csv(self._config["inputs"]["future_covariates"]["common"]["model_data"]["test"])
            self._input_future_predict["day"] = self._input_future_predict["day"].astype(str)
            self._input_future_test["day"] = self._input_future_test["day"].astype(str)
            self._input_future_train["day"] = self._input_future_train["day"].astype(str)
            self._input_future_predict["month"] = self._input_future_predict["month"].astype(str)
            self._input_future_train["month"] = self._input_future_train["month"].astype(str)
            self._input_future_test["month"] = self._input_future_test["month"].astype(str)
        else :
            self._categorical_cols = []
            self._input_future_predict = pd.DataFrame(columns=['time'])
            self._input_future_train = pd.DataFrame(columns=['time'])
            self._input_future_test = pd.DataFrame(columns=['time'])
        self._input_past_predict = pd.read_csv("resources/input/model_data/input_past_predict.csv")
        self._input_past_train = pd.read_csv("resources/input/model_data/input_past_train.csv")
        self._output_predict = pd.read_csv("resources/input/model_data/output_predict.csv")
        self._output_train = pd.read_csv("resources/input/model_data/output_train.csv")

        self._input_past_test = pd.read_csv("resources/input/model_data/input_past_test.csv")
        self._output_test = pd.read_csv("resources/input/model_data/output_test.csv")

        self._input_past_train.columns = self._input_past_train.columns.str.replace('.', '_', regex=True)
        self._input_past_predict.columns = self._input_past_predict.columns.str.replace('.', '_', regex=True)
        self._input_past_test.columns = self._input_past_test.columns.str.replace('.','_',regex=True)




    def _obtain_dataloader(self):
        self._train_data = pd.concat([self._input_past_train, self._input_future_train.drop(columns=['time']), self._output_train[['return']]], axis=1)
        self._val_data = pd.concat([self._input_past_predict, self._input_future_predict.drop(columns=['time']), self._output_predict[['return']]], axis=1)
        self._test_data = pd.concat([self._input_past_test, self._input_future_test.drop(columns=['time']), self._output_test[['return']]], axis=1)

        self._train_data['group'] = 'group_1'
        self._val_data['group'] = 'group_1'
        self._test_data['group'] = 'group_1'

        self._continuous_cols = [col for col in self._input_past_train.columns if col not in ["time"]]
        if ("RecurrentNetwork" or "DeepAR") not in self._model_name:
            self._continuous_cols.append("return")
            encoder_variables = []

        # else :
        #     self._training_dataset = TimeSeriesDataSet(
        #         self._train_data,
        #         time_idx="time",
        #         target="return",
        #         group_ids=["group"],
        #         max_encoder_length=self._params["max_encoder_length"],
        #         max_prediction_length=self._params["max_prediction_length"],
        #         time_varying_known_categoricals=self._categorical_cols,
        #         time_varying_unknown_reals=self._continuous_cols,
        #     )
        #
        #     encoder_variables = set(
        #         self._training_dataset.time_varying_known_reals + self._training_dataset.time_varying_unknown_reals)
        #     encoder_variables.discard('return')
        #     self._continuous_cols = encoder_variables


        self._training_dataset = TimeSeriesDataSet(
            self._train_data,
            time_idx="time",
            target="return",
            group_ids=["group"],
            max_encoder_length=self._params["max_encoder_length"],
            max_prediction_length=self._params["max_prediction_length"],
            #time_varying_known_categoricals=self._categorical_cols,
            time_varying_unknown_reals=self._continuous_cols,
            #time_varying_known_reals=encoder_variables,
        )

        print("Time Varying Known Reals:",
              self._training_dataset.time_varying_known_reals)
        print("Time Varying Unknown Reals:",
              self._training_dataset.time_varying_unknown_reals)

        encoder_variables = set(
            self._training_dataset.time_varying_known_reals + self._training_dataset.time_varying_unknown_reals)
        #encoder_variables.discard('return')

        decoder_variables = set(
            self._training_dataset.time_varying_known_reals + self._training_dataset.time_varying_unknown_reals)

        assert encoder_variables == decoder_variables, "Mismatch between encoder and decoder variables"


        self._train_dataloader = self._training_dataset.to_dataloader(train=True, batch_size=self._params['batch_size'], num_workers=0)
        self._val_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, self._val_data, predict=False, stop_randomization=True)
        self._val_dataloader = self._val_dataset.to_dataloader(train=False, batch_size=self._params['batch_size']*1000, num_workers=0)
        self._test_dataset = TimeSeriesDataSet.from_dataset(self._training_dataset, self._test_data, predict=False, stop_randomization=True)
        self._test_dataloader = self._test_dataset.to_dataloader(train=False, batch_size=self._params['batch_size']*1000, num_workers=0)

    def _train_model(self):
        pl.seed_everything(self._params['random_state'])

        model_to_train = self._config_manager.get_custom_model(self._model_name)

        self._model = RecurrentNetwork.from_dataset(
            dataset=self._training_dataset,
            **self._config_manager.models_hyperparameters[self._model_name],
        )
        logger = TensorBoardLogger("lightning_logs")

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

        logger = TensorBoardLogger("lightning_logs", name= self._model_name)
        trainer = pl.Trainer(max_epochs=self._params["epochs"],
                             callbacks=callbacks_list,
                             logger=logger,
                             gradient_clip_val=self._params["gradient_clip_val"],
                             enable_model_summary=True,
                             )

        trainer.fit(self._model,
                    train_dataloaders=self._train_dataloader,
                    val_dataloaders=self._val_dataloader)

    def _calculate_metrics(self,dataloader, data):
        self._all_predicted_returns = []
        self._all_actual_returns = []
        self._time_indices = []
        self._cumulative_predicted_return = 1
        self._cumulative_actual_return = 1
        self._cumulative_max_possible = 1
        self._cumulative_index = len(data) - len(
            dataloader.dataset)

        predictions = self._best_model.predict(dataloader, mode ="raw",  return_x=True,return_y = True)
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
        self._all_predicted_returns = [t.item() for t in self._all_predicted_returns]

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
            f'models/{self._model_name}', 'metrics.json')
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
        best_model_path = self._model_checkpoint.best_model_path
        self._best_model = self._model.load_from_checkpoint(best_model_path)
        #self._best_model = self._model.load_from_checkpoint('models/pytorch_forecasting_local/TemporalFusionTransformer/best_model-TemporalFusionTransformer-epoch=00-val_loss=0.00.ckpt')

        for dataloader, data, data_type in [(self._val_dataloader, self._val_data, 'val'), (self._test_dataloader, self._test_data, 'test')]:
            if dataloader is None or data is None:
                continue

            self._calculate_metrics( dataloader, data)
            self._process_metrics(data)
            self._save_metrics(data_type)
            self._plot_predictions(data_type)