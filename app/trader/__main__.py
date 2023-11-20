import os
from app.shared.data_processor import (
    DataProcessorHelper
)
from app.shared.config_utils import InitProject, ConfigManager
from app.shared.factory import DataSourceFactory


if __name__ == "__main__":
    os.environ['ORIGINAL_SCRIPT'] = __file__
    config_manager = ConfigManager()
    InitProject.create_common_path()
    InitProject.create_custom_path()

    data_processor_helper = DataProcessorHelper(config_manager=config_manager)
    for source, is_input in config_manager.get_sources():
        data_for_source = config_manager.get_config_for_source(source, is_input)

        data_source = DataSourceFactory.implement_data_source(
            data_for_source,
            data_processor_helper=data_processor_helper,
            is_input_feature=is_input,
        ).run()
