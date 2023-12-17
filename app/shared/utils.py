import pandas_market_calendars as mcal
from datetime import datetime, timedelta
from pytz import timezone, UTC
from functools import lru_cache
import json
import pandas as pd
import shutil
import threading

from app.shared.config.constants import (
    RAW_ATTRIBUTES,
)

from typing import Optional
import os
import numpy as np

def numpy_to_python(data):
    if isinstance(data, dict):
        return {k: numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data

def save_json(file: str, data: dict) -> None:
    data = numpy_to_python(data)
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(file: str) -> dict:
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def create_file_if_not_exist(file: str) -> None:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8"):
            pass


def string_to_datetime(string_to_convert: str) -> datetime:
    return datetime.strptime(string_to_convert, "%Y-%m-%d")


def utc_to_est_str(utc_timestamp: int) -> str:
    utc_dt = datetime.utcfromtimestamp(utc_timestamp)
    est_tz = timezone("US/Eastern")
    est_dt = utc_dt.astimezone(est_tz)
    return est_dt.strftime("%Y-%m-%d")


def est_dt_to_epoch(dt: datetime) -> int:
    est = timezone("US/Eastern")
    localized_dt = est.localize(dt)
    utc_dt = localized_dt.astimezone(UTC)
    return int(utc_dt.timestamp())


def obtain_market_dates(start_date: str, end_date: str, market : Optional[str] = "NYSE") -> pd.DataFrame:
    nyse = mcal.get_calendar(market)
    market_open_dates = nyse.schedule(
        start_date=start_date,
        end_date=end_date,
    )
    return market_open_dates

def add_days_to_date(original_date, days : int = 0) -> str:
    date_object = datetime.strptime(original_date, "%Y-%m-%d")
    new_date_object = date_object + timedelta(days=days)
    return new_date_object.strftime("%Y-%m-%d")


@lru_cache(maxsize=None)
def get_previous_market_date(date, last_market_date=None, market : Optional[str]= "NYSE"):
    nyse = mcal.get_calendar(market)
    if last_market_date is None:
        start_date = date - timedelta(days=30)
    else:
        start_date = last_market_date
    market_open_dates = nyse.valid_days(start_date=start_date, end_date=date)
    if market_open_dates.empty:
        raise ValueError(
            f"No valid NYSE market date found within the previous 60 days from the given date {date}"
        )
    last_market_date = market_open_dates[-1].date()
    return last_market_date, market_open_dates[-1].date()


def _replace_for_sliding_window(file : str, window : Optional[int] = '') -> str:
    if window or window ==0 :
        file = file.replace('.csv', f'_{window}.csv')
    return file

def read_csv_to_pd_formatted(
    file: str,
    sort_by_column_name: Optional[str] = RAW_ATTRIBUTES[0],
        window : Optional[int] = ''
) -> pd.DataFrame:
    file = _replace_for_sliding_window(file, window)

    pd_data = pd.read_csv(file, encoding="utf-8")
    pd_data = pd_data.sort_values(by=sort_by_column_name, ascending=True)
    pd_data = pd_data.reset_index(drop=True)
    return pd_data

def write_pd_to_csv(data:pd.DataFrame, file: str,
                    sort_by_column_name: Optional[str] = RAW_ATTRIBUTES[0]) -> None:
    data = data.reset_index(drop=True)
    data = data.sort_values(by=sort_by_column_name, ascending=True)
    data.to_csv(file, index=False, encoding="utf-8")


def write_to_csv_formatted(
    data: pd.DataFrame, file: str, sort_by_column_name: Optional[str] = RAW_ATTRIBUTES[0],
        window : Optional[int] = ''
) -> None:
    file = _replace_for_sliding_window(file, window)

    _raise_error_if_nan_value(data)
    write_pd_to_csv(data,file,sort_by_column_name)


def _raise_error_if_nan_value(data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError(
            "DataFrame contains empty values at row(s): "
            + str(list(data.index))
        )

    if data.isna().sum().sum() > 0:
        na_rows = data.index[data.isna().any(axis=1)].tolist()
        raise ValueError(
            "DataFrame contains NaN values at row(s): " + str(na_rows)
        )

    if (data == ".").sum().sum() > 0:
        dot_rows = data.index[(data == ".").any(axis=1)].tolist()
        raise ValueError(
            'DataFrame contains values that are exactly equal to "." at row(s): '
            + str(dot_rows)
        )

def clear_directory_content(directory_path : str, exclusions : Optional[int] = None) -> None:
    if exclusions is None:
        exclusions = []
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            if filename in exclusions:
                continue
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def play_music():
    music_thread = threading.Thread(
    target=os.system('afplay super-mario-bros.mp3'))
    music_thread.start()
