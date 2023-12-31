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
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################
from app.shared.config.config_utils import InitProject, ConfigManager
from app.shared.data_processor import (
    DataForModelSelector,
    FutureCovariatesProcessor,
    DataProcessorHelper
)


from app.shared.factory import DataSourceFactory
from app.trainer.models.models import HyperpametersOptimizer, ModelBuilder
import threading
import torch
import os


if __name__ == "__main__":
    config_manager = ConfigManager(file='app/trainer/config.yaml')
    InitProject.create_common_path()
    InitProject.create_custom_path(file='app/trainer/config.yaml')

    data_processor_helper = DataProcessorHelper(config_manager=config_manager)
    for source, is_input in config_manager.get_sources():
        data_for_source = config_manager.get_config_for_source(source, is_input)

        data_source = DataSourceFactory.implement_data_source(
            data_for_source,
            data_processor_helper=data_processor_helper,
            is_input_feature=is_input,
        ).run()
    if config_manager.config["inputs"]["future_covariates"]["data"]:
        for window in range(config_manager.config['common']['sliding_windows']):
            FutureCovariatesProcessor(
                config_manager, data_processor_helper, sliding_window=window
            ).run()
    if config_manager.config['common']['engineering']:
        DataForModelSelector(config_manager).run()
    if config_manager.config['common']['hyperparameters_optimization']['is_optimizing']:
            HyperpametersOptimizer(config_manager).run()

    ModelBuilder(config_manager).run()

    print('Program finished successfully')
    if not torch.cuda.is_available():
        music_thread = threading.Thread(
            target=os.system('afplay super-mario-bros.mp3'))
        music_thread.start()


