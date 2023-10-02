from typing import Optional

from mean_reversion.data_processor import (
    BaseDataProcessor,
    AlphaVantage,
    FRED,
    MacroTrends
)


class DataSourceFactory:
    @classmethod
    def implement_data_source(
        cls,
        data_for_source: dict,
        data_processor_helper,
        is_input_feature: Optional[bool] = True,
    ) -> BaseDataProcessor:
        source = data_for_source.get("source").lower()
        if source == "reddit":
            return Reddit(
                specific_config=data_for_source, data_processor_helper=data_processor_helper, is_input_feature=is_input_feature
            )
        if source == "av":
            return AlphaVantage(
                specific_config=data_for_source, data_processor_helper=data_processor_helper, is_input_feature=is_input_feature
            )
        elif source == "fred":
            return FRED(
                specific_config=data_for_source, data_processor_helper=data_processor_helper, is_input_feature=is_input_feature
            )
        elif source == "macrotrends":
            return MacroTrends(
                specific_config=data_for_source, data_processor_helper=data_processor_helper, is_input_feature=is_input_feature
            )

        else:
            raise ValueError(f"Unknown data type: {source}")
