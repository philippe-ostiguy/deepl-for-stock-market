from mean_reversion.utils import read_json, obtain_market_dates, clear_directory_content
from mean_reversion.config.constants import (
    RAW_PATH,
    PREPROCESSED_PATH,
    ENGINEERED_PATH,
    MODEL_PHASES,
    MODEL_DATA_PATH,
    MODELS_PATH,
    TRANSFORMATION_PATH,
    PATHS_TO_CREATE
)

from mean_reversion.config.model_config import (
    PYTORCH_CALLBACKS,
    LOSS,
    MODEL_MAPPING,
    CUSTOM_MODEL

)


from copy import deepcopy
from typing import List, Dict, Text, Any, Optional, Tuple, Union
import os
import yaml


class ConfigManager:
    def __init__(self, file: Optional[Text] = "config.yaml") -> None:
        self._config = self._load_config(file)
        self._add_ticker_dynamically()
        self._assign_inputs()
        self._config['specific_config'] = {}
        self._assign_common_config()
        self._config["output"] = self._add_data_files_output(
            self._config["output"]
        )

        self._models_with_hyper = read_json(
            "resources/configs/models_args.json"
        )
        self._models_support = read_json(
            "resources/configs/models_support.json"
        )
        self.models_hyperparameters = {}
        self._models_with_hyperparameters_to_optimize = {}
        self._assign_hyperparameters_to_models()
        self._validate_num_forecasts()
        self._validate_model_metrics()
        self._config["common"]["best_model_path"] = \
            os.path.join(MODELS_PATH, "best_model")
        if self._config["common"]['preprocessing']:
            clear_directory_content(PREPROCESSED_PATH)
        if self._config["common"]['engineering']:
            clear_directory_content(ENGINEERED_PATH)
            clear_directory_content(MODEL_DATA_PATH)

    def _assign_common_config(self):
        common_config = self._config["hyperparameters"]["common"]
        if not "gradient_clip_val" in common_config:
            self._config["hyperparameters"]["common"]["gradient_clip_val"] = None

    def _add_ticker_dynamically(self):
        for sub_dict in self._config["inputs"]["past_covariates"]:
            data_items = sub_dict.get('data', [])
            for item in data_items:
                if "set_dynamically" in item:
                    sub_dict["data"].remove(item)
                    subname = item.replace('set_dynamically', '')
                    dynamic_data = self._load_data_dynamically(sub_dict["source"], subname)
                    sub_dict["data"].extend(dynamic_data)
            sub_dict["data"] = list(set(sub_dict["data"]))

    def _load_data_dynamically(self, source: str, subname : Optional[str] ='') -> List[str]:
        file_path = f"resources/configs/dynamic_ticker/{source}{subname}_daily.txt"

        with open(file_path, "r") as file:
            return [line.strip() for line in file]

    def get_custom_model(self, model :str):
        return CUSTOM_MODEL[model]

    def get_loss(self, model : str) -> Union[dict,None]:
        if self._check_if_argument_exist(model,'loss'):
            return deepcopy(self.models_hyperparameters[model]['loss'])
        return None

    def _check_if_argument_exist(self, model : str, argument_to_check : str) -> bool:
        if self.models_hyperparameters and argument_to_check in self.models_hyperparameters[model]:
            return True
        return False

    def _assign_callbacks(self, model):
        pl_trainer_kwargs = deepcopy(self._config["hyperparameters"]["common"]["pl_trainer_kwargs"])
        callback_config = deepcopy(self._config["hyperparameters"]["common"]["callbacks"])
        if not pl_trainer_kwargs:
            del self._config["hyperparameters"]["common"]["pl_trainer_kwargs"]
            return

        for pl_trainer_argument, pl_trainer_value in pl_trainer_kwargs.items():
            if isinstance(pl_trainer_value, list):
                pl_trainer_kwargs[pl_trainer_argument] = []
                for callback_name in pl_trainer_value:
                    if callback_name in callback_config:
                        callback_args = callback_config[callback_name]

                        if callback_name == "EarlyStopping":
                            monitor_value = callback_args.get('monitor')
                            if monitor_value == 'val_PortfolioReturnMetric':
                                callback_args[
                                    'monitor'] = monitor_value

                                model_checkpoint_args = callback_config.get(
                                    "ModelCheckPoint", {})
                                model_checkpoint_args['monitor'] = monitor_value

                        if callback_name == "ModelCheckPoint":
                            if callback_args.get(
                                    'monitor') == 'val_PortfolioReturnMetric':
                                callback_args['mode'] = 'max'
                            if callback_args.get(
                                    'monitor') == 'val_PortfolioReturnMetric':
                                callback_args[
                                    'dirpath'] = f'{MODELS_PATH}/{model}'

                        callback_instance = PYTORCH_CALLBACKS[callback_name](
                            **callback_args)
                        pl_trainer_kwargs[pl_trainer_argument].append(
                            callback_instance)
        return pl_trainer_kwargs

    def _validate_model_metrics(self):
        for metric in self._config["common"]["metrics_to_choose_model"]:
            if metric not in self._models_support["supported_metrics"]:
                raise ValueError(
                    f"Metric {metric} is not a supported metric to"
                    f"pick the best model"
                )

    def _validate_num_forecasts(self):
        total_days = len(
            obtain_market_dates(
                self._config["common"]["start_date"],
                self._config["common"]["end_date"],
            )
        )

        input_length = self._config["hyperparameters"]["common"][
            "max_encoder_length"
        ]
        output_length = self._config["hyperparameters"]["common"][
            "max_prediction_length"
        ]
        train_test_split = self._config["common"]["train_test_split"]
        validation_forecasts = (
            total_days * (train_test_split[1])
            - input_length
            - output_length
            + 1
        )
        min_forecasts = self._config["common"]["min_validation_forecasts"]

        if validation_forecasts < min_forecasts:
            raise ValueError(
                f"Not enough forecasts for validation,"
                f"minimum is {min_forecasts} but got {validation_forecasts} \n"
                f"In configuration, decrease input_chunk_length, decrease output_chunk_length,"
                f"increase the date range and/or decrease train_test_split"
            )

    def get_model_args(self, model_name: str, model_argument_type: str,
                       keys_only: bool = False) -> dict:
        model_args = self._models_with_hyper.get(model_name)[
            model_argument_type]
        return model_args.keys() if keys_only else model_args

    def _assign_inputs(self):
        tempo_inputs = self._config["inputs"]["past_covariates"]
        self._config["inputs"]["past_covariates"] = {}
        self._config["inputs"]["past_covariates"]["sources"] = tempo_inputs
        self._config["inputs"]["past_covariates"]["common"] = {}
        self._assign_common_inputs()
        self._config["inputs"]["past_covariates"][
            "common"
        ] = self._add_model_data_path(
            self._config["inputs"]["past_covariates"]["common"], "input_past"
        )

        self._config["inputs"]["past_covariates"][
            "sources"
        ] = self._add_data_files_input(
            self._config["inputs"]["past_covariates"]["sources"]
        )

        self._config["inputs"]["future_covariates"]["common"] = {}
        self._config["inputs"]["future_covariates"][
            "common"
        ] = self._add_model_data_path(
            self._config["inputs"]["future_covariates"]["common"],
            "input_future",
        )


    def _assign_common_inputs(self):
        self._config["inputs"]["past_covariates"]["common"]["pickle"] = {}
        self._config["inputs"]["past_covariates"]["common"]["pickle"][
            "features_to_keep"
        ] = f"{TRANSFORMATION_PATH}/features_to_keep.pickle"
        self._config["inputs"]["past_covariates"]["common"]["pickle"][
            "transformations_for_stationarity"
        ] = f"{TRANSFORMATION_PATH}/transformations_for_stationarity.pickle"
        self._config["inputs"]["past_covariates"]["common"]["pickle"][
            "transformations_for_scaler"
        ] = f"{TRANSFORMATION_PATH}/transformations_for_scaler.pickle"

    def _assign_hyperparameters_to_models(self):

        for model in self._models_with_hyper:
            if model == 'common':
                continue
            self._config['specific_config'][model] = {}
            self._config['specific_config'][model] = deepcopy(self._assign_callbacks(model))
            model_args = self._assign_model_hyperparameters(model)
            self.models_hyperparameters[
                model] = model_args

            if self._config["common"]["hyperparameters_optimization"]["is_optimizing"]:
                self._models_with_hyperparameters_to_optimize[
                    model] = self._assign_optimization_hyperparameters(model,
                                                                       model_args)

    def _assign_model_hyperparameters(self, model):
        common_config = deepcopy(self._config["hyperparameters"]["common"])
        model_specific = deepcopy(self._config["hyperparameters"]["models"].get(model,
                                                                       {}))
        model_keys = self._models_with_hyper[model]["hyperparameters"]

        model_args = {}
        model_args.update(
            self._extract_config_values(model_keys.keys(), model_specific))
        likelihood=  []

        if "likelihood" in common_config and isinstance(common_config["likelihood"],
                                                     list):
            likelihood = common_config["likelihood"]


        if "loss" in model_args and ('Quantile' in model_args["loss"] or 'DistributionLoss' in model_args["loss"]):
            if likelihood:
                model_args['loss'] = LOSS[model_args['loss']](quantiles = likelihood)

            else:
                model_args['loss'] = LOSS[model_args['loss']]()
        elif "loss" in model_args:
            model_args['loss'] = LOSS[model_args['loss']]()

        return model_args

    def _extract_config_values(self, keys : str, config : dict) -> dict:
        return {k: config[k] for k in keys if k in config}

    def _assign_optimization_hyperparameters(self, model : str, default_args : dict) -> dict:
        common_optimization_config = \
        self._config["hyperparameters_optimization"]["common"]
        model_specific_optimization_config = \
        self._config["hyperparameters_optimization"]["models"].get(model, {})
        model_keys = self._models_with_hyper[model]["hyperparameters"]

        optimization_args = self._extract_config_values(model_keys.keys(),
                                                        common_optimization_config)
        optimization_args.update(self._extract_config_values(model_keys.keys(),
                                                             model_specific_optimization_config))

        missing_keys = set(default_args.keys()) - set(optimization_args.keys())
        optimization_args.update({k: default_args[k] for k in missing_keys})

        return optimization_args

    def get_model_hyperparameters(self, model_name: str) -> dict:
        return self.models_hyperparameters.get(model_name, None)

    def get_model_hyperparameters_to_optimize(self, model_name: str) -> dict:
        return self._models_with_hyperparameters_to_optimize.get(model_name,
                                                                 None)

    def _load_config(self, file: Text) -> dict:
        with open(file, "r") as yaml_file:
            configuration = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self._validate_configuration(configuration)
        configuration = self._settle_configuration(configuration)

        return configuration

    def _validate_configuration(self, config: dict) -> None:
        model_phase = config["common"]["model_phase"]
        is_error = True
        for phase in MODEL_PHASES:
            if model_phase == phase:
                is_error = False
                break
        if is_error:
            raise ValueError(f"model_phase key must be either train or predict")

    def _settle_configuration(self, configuration: dict) -> dict:
        common = configuration["common"]
        if (
            "max_missing_data" not in common
            or not isinstance(common["max_missing_data"], float)
            or not 0 <= common["max_missing_data"] <= 1
        ):
            configuration["common"]["max_missing_data"] = 0.01
        return configuration

    def _add_data_files_input(
        self, data_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        for sub_dict in data_sources:
            new_data = []
            for value in sub_dict["data"]:
                new_data.append(self._add_path_values_in_config(value, "input"))
            sub_dict["data"] = new_data
        return data_sources

    def _add_data_files_output(self, output_data: Dict) -> Dict:
        value = output_data["data"]
        output_data = self._add_model_data_path(output_data, "output")
        output_data["data"] = [self._add_path_values_in_config(value, "output")]
        return output_data

    def _add_model_data_path(self, sub_dict: Dict, data_type: str) -> Dict:
        sub_dict["model_data"] = {}
        sub_dict["model_data"]["train"] = os.path.join(
            MODEL_DATA_PATH, f"{data_type}_train.csv"
        )
        sub_dict["model_data"]["predict"] = os.path.join(
            MODEL_DATA_PATH, f"{data_type}_predict.csv"
        )
        sub_dict["model_data"]["test"] = os.path.join(
            MODEL_DATA_PATH, f"{data_type}_test.csv"
        )
        sub_dict["model_data"]["scaler"] = os.path.join(
            MODEL_DATA_PATH, f"{data_type}_scaler.pkl"
        )
        return sub_dict

    def _add_path_values_in_config(self, value: str, data_type: str) -> Dict:
        return {
            "asset": value,
            "raw": os.path.join(RAW_PATH, f"{value}_{data_type}.csv"),
            "preprocessed": os.path.join(
                PREPROCESSED_PATH, f"{value}_{data_type}.csv"
            ),
            "engineered": {
                "train": os.path.join(
                    ENGINEERED_PATH, f"{value}_{data_type}_train.csv"
                ),
                "predict": os.path.join(
                    ENGINEERED_PATH, f"{value}_{data_type}_predict.csv"
                ),
                "test": os.path.join(
                    ENGINEERED_PATH, f"{value}_{data_type}_test.csv"
                ),
            },
        }

    def _obtain_data_for_current_source(self, current_source: str) -> Dict:
        return next(
            (
                item
                for item in self._config["inputs"]["past_covariates"]["sources"]
                if item["source"] == current_source
            ),
            None,
        )

    @property
    def config(self) -> dict:
        return self._config

    def get_config_for_source(self, source: str, is_input: bool) -> dict:
        if is_input:
            data_for_source = self._obtain_data_for_current_source(source)
            if not data_for_source:
                raise ValueError(
                    f"No configuration found for input source: {source}"
                )
            data_for_source_common = deepcopy(self._config["common"])
            past_covariates_common = deepcopy(
                self._config["inputs"]["past_covariates"]["common"]
            )
            common_keys = set(data_for_source_common.keys()) & set(
                past_covariates_common.keys()
            )

            if common_keys:
                raise ValueError(
                    f"Duplicated keys found in configurations: {common_keys}"
                )
            data_for_source["common"] = {
                **data_for_source_common,
                **past_covariates_common,
            }
            return data_for_source

        else:
            if self._config["output"]["source"] == source:
                output_config = self._config["output"]
                output_config["common"] = self._config["common"]
                return output_config
            else:
                raise ValueError(
                    f"No configuration found for output source: {source}"
                )

    def get_sources(self) -> List[Tuple[str, bool]]:
        input_sources = [
            (src["source"], True)
            for src in self._config["inputs"]["past_covariates"]["sources"]
        ]
        output_sources = [(self._config["output"]["source"], False)]
        return input_sources + output_sources


class InitProject:
    def __init__(self, paths_to_create: list = PATHS_TO_CREATE, config : Optional[ConfigManager] = None):
        self._paths_to_create = paths_to_create
        self._config = config


    @classmethod
    def create_custom_path(cls, config) -> Any:
        paths_to_create = []
        for model in config.config['hyperparameters']['models']:
            paths_to_create.append(os.path.join(MODELS_PATH,model))
        cls._create_path(paths_to_create)


    @classmethod
    def create_common_path(cls,paths_to_create : list = PATHS_TO_CREATE):
        cls._create_path(paths_to_create)

    @classmethod
    def _create_path(self, paths_to_create :list):
        for path_to_create in paths_to_create:
            if not os.path.exists(path_to_create):
                os.makedirs(path_to_create)