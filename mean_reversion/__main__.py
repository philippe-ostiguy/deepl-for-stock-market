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

from mean_reversion.data_processor import (
    DataForModelSelector,
    FutureCovariatesProcessor,
    DataProcessorHelper
)

from mean_reversion.config.config_utils import InitProject, ConfigManager
from mean_reversion.factory import DataSourceFactory
from mean_reversion.models.models import ModelBuilder, HyperpametersOptimizer
import torch
torch.manual_seed(42)

if __name__ == "__main__":

    init_project = InitProject()
    init_project.create_common_path()
    config_manager = ConfigManager()
    init_project.create_custom_path(config_manager)

    data_processor_helper = DataProcessorHelper(config_manager=config_manager)
    for source, is_input in config_manager.get_sources():
        data_for_source = config_manager.get_config_for_source(source, is_input)

        data_source = DataSourceFactory.implement_data_source(
            data_for_source,
            data_processor_helper=data_processor_helper,
            is_input_feature=is_input,
        ).run()
    if config_manager.config["inputs"]["future_covariates"]["data"]:
        FutureCovariatesProcessor(
            config_manager, data_processor_helper
        ).run()
    DataForModelSelector(config_manager).run()
    if config_manager.config['common']['hyperparameters_optimization']['is_optimizing']:
           HyperpametersOptimizer(config_manager).run()

    # ModelBuilder(config_manager).run()

