from typing import Optional

from app.shared.data_processor import (
    BaseDataProcessor,
    YahooFinance,
    FRED)


class DataSourceFactory:
    @classmethod
    def implement_data_source(
        cls,
        data_for_source: dict,
        config_manager,
        data_processor_helper,
        is_input_feature: Optional[bool] = True,
    ) -> BaseDataProcessor:
        source = data_for_source.get("source").lower()
        if source == "yahoo":
            return YahooFinance(
                specific_config=data_for_source, config_manager=config_manager, data_processor_helper=data_processor_helper, is_input_feature=is_input_feature
            )

        if source == "fred":
            return FRED(
                specific_config=data_for_source,config_manager=config_manager, data_processor_helper=data_processor_helper, is_input_feature=is_input_feature
            )
        else:
            raise ValueError(f"Unknown data type: {source}")
