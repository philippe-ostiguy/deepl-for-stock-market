from app.shared.config.config_utils import ConfigManager, singleton
from typing import Optional

@singleton
class ModelValueRetriver:
    def __init__(self, config_manager : Optional[ConfigManager] = None):
        if config_manager is None:
            config_manager = ConfigManager(file='app/trainer/config.yaml')
        self._config = config_manager.config
        self._confidence_indexes = ''

    @property
    def confidence_indexes(self):
        if self._confidence_indexes:
            return self._confidence_indexes
        confidence_level = \
            self._config["hyperparameters"]["common"][
                "confidence_level"] if "confidence_level" in \
                                       self._config["hyperparameters"][
                                           "common"] else .5
        likelihood = self._config["hyperparameters"]["common"][
            "likelihood"]
        upper_index = likelihood.index(confidence_level)
        lower_index = len(likelihood) - 1 - upper_index
        return lower_index, upper_index

    @confidence_indexes.setter
    def confidence_indexes(self, values: tuple):
        self._confidence_indexes = values
