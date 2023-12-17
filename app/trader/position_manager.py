from ib_insync import IB, Stock, MarketOrder, OrderCondition,Contract, Order, TimeCondition, StopOrder
import os
from datetime import datetime, timedelta
import pytz
from app.shared.utils import play_music

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
            self._ib.connect('127.0.0.1', 7497, clientId=1)
            self._get_position_size()
            self._enter_market()
            #self._enter_stop_loss()
            self._ib.disconnect()


    def _check_for_open_positions(self):
        positions = self._ib.positions()
        if positions:
            play_music()
            raise RuntimeError("Open positions exist.")

    def _enter_market(self):
        self._check_for_open_positions()
        trade_details = Contract()
        trade_details.symbol = "MCL"
        trade_details.secType = "FUT"
        trade_details.exchange = "NYMEX"
        trade_details.currency = "USD"
        trade_details.lastTradeDateOrContractMonth = "20231218"

        trades = self._ib.fills()
        filtered_trades = [trade for trade in trades if trade.contract.symbol == trade_details.symbol and
                           trade.contract.secType == trade_details.secType and
                           trade.contract.exchange == trade_details.exchange and
                           trade.contract.currency == trade_details.currency and
                           trade.contract.lastTradeDateOrContractMonth == trade_details.lastTradeDateOrContractMonth]
        self._trade = filtered_trades[0] if filtered_trades else None

        #self._open_positions = 123
        #trade_details = Stock('CEG', 'SMART', 'USD')


        if not self._trade:
            main_order = MarketOrder('BUY', 1)
            self._trade = self._ib.placeOrder(trade_details, main_order)
            # time_condition = TimeCondition()
            # time_condition.time = '09:45:00 EST'
            # time_condition.isMore = True
            # main_order.conditions.append(time_condition)
            self._trade = self._ib.placeOrder(trade_details, main_order)
            while self._trade.orderStatus.status not in ['Filled', 'Cancelled', 'Rejected']:
                self._ib.sleep(5)

            if self._trade.orderStatus.status != 'Filled':
                play_music()
                raise Exception(f"Order not filled, status: {self._trade.orderStatus.status}")


        oca_group_name = "OCA_" + str(self._trade.order.orderId)
        stop_loss_price = self._trade.fills[0].execution.price *(1-self._asset_details[self._index]['stop_loss'])
        stop_loss_price = round(stop_loss_price,2)
        stop_loss_order = StopOrder('SELL',1,stop_loss_price)
        stop_loss_order.ocaGroup = oca_group_name

        sell_market_order = MarketOrder('SELL', 1)
        sell_market_order.ocaGroup = oca_group_name
        sell_market_order.orderType = 'MKT'
        sell_market_order.tif = 'GTC'
        time_condition = TimeCondition()
        time_condition.isMore = True
        time_condition.time = '15:30:00 EST'
        sell_market_order.conditions.append(time_condition)

        self._ib.placeOrder(trade_details, stop_loss_order)
        self._ib.placeOrder(trade_details, sell_market_order)


    def _enter_stop_loss(self):
        self._ib.connect('127.0.0.1', 7497, clientId=1)
        stock = Stock('AAPL', 'SMART', 'USD')
        bars = self._ib.reqHistoricalData(
            stock,
            endDateTime='',
            durationStr='120 D',
            barSizeSetting='1 D',
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
        print(f'{asset} position :{position_size} with stop loss :  {stop_loss} and {nb_shares_round} shares. Last '
              f'close was {data["close"].iloc[-1]}')
        t = 5
