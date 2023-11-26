import time

from app.shared.config.config_utils import ConfigManager
from app.shared.utils import read_csv_to_pd_formatted
from scipy.stats import norm
from ib_insync import IB, Stock, MarketOrder

class PositionManager:
    def __init__(self, data_for_source,config_manager : ConfigManager):
        self._config_manager = config_manager
        self._config = data_for_source
        self._asset_details = []
        self._ib = IB()

    def run(self):

        for index, _ in enumerate(self._config["data"]):
            self._index = index
            self._input_file = self._config["data"][index]["preprocessed"]
            self._asset_details.append({'asset' : self._config['data'][index]['asset']})
            self._get_position_size()
            self._enter_market()
            self._enter_stop_loss()

    def _enter_market(self):
        self._ib.connect('127.0.0.1', 7497, clientId=1)
        stock = Stock('AAPL', 'SMART', 'USD')
        order = MarketOrder('BUY', 1)
        self._trade = self._ib.placeOrder(stock, order)
        time.sleep(10)
        self._ib.disconnect()

    def _enter_stop_loss(self):
        self._ib.connect('127.0.0.1', 7497, clientId=1)
        stock = Stock('AAPL', 'SMART', 'USD')
        bars = self._ib.reqHistoricalData(
            stock,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        if bars:
            latest_trade = bars[-1]
            print(f"Date/Time: {latest_trade.date}, Open: {latest_trade.open}, "
                  f"High: {latest_trade.high}, Low: {latest_trade.low}, Close: {latest_trade.close}")
        else:
            print("No data received.")

        self._ib.disconnect()


    def _get_position_size(self):
        asset = self._asset_details[self._index]["asset"]
        self._pos_management = self._config_manager.config['position_management']
        data  = read_csv_to_pd_formatted(self._input_file)
        data = data[-120:]
        data = data.copy()
        data['return'] = data['close'] / data['open'] - 1
        std_dev = data['return'].std()
        if std_dev == 0:
            raise ValueError(f'std deviation is 0 for {asset}')

        proportion =self._pos_management['quantile'] / 100
        z_score = norm.ppf((1 + proportion) / 2)
        position_size = (self._pos_management['trading_capital']) \
                        / ((z_score* std_dev)/self._pos_management['return_in_quantile'])
        nb_shares = position_size/data['close'].iloc[-1]
        nb_shares_round = int(nb_shares//5)*5
        stop_loss = self._pos_management['trading_capital']*self._pos_management['risk_per_trade']/position_size
        self._asset_details[self._index]['position_size'] = position_size
        self._asset_details[self._index]['stop_loss'] = stop_loss
        self._asset_details[self._index]['nb_shares'] = nb_shares_round
        print(f'{asset} position :{position_size} with stop loss :  {stop_loss}')
