import os

from app.shared.config.config_utils import InitProject, ConfigManager
from app.shared.factory import DataSourceFactory
from app.trader.position_manager import PositionManager

if __name__ == "__main__":
    os.environ['ORIGINAL_SCRIPT'] = __file__
    config_manager = ConfigManager(file='app/trader/config.yaml')
    InitProject.create_common_path()

    for source, is_input in config_manager.get_sources():
        data_for_source = config_manager.get_config_for_source(source, is_input)

        data_source = DataSourceFactory.implement_data_source(
            data_for_source,
            config_manager,
            data_processor_helper=None,
            is_input_feature=is_input,
        ).run()
        position_manager = PositionManager(data_for_source =data_for_source,config_manager=config_manager)
        position_manager.run()
