from datamodel import OrderDepth, TradingState, Order, Symbol, Product, ConversionObservation
from collections import deque
from typing import List, Dict
from enum import IntEnum
from statistics import NormalDist
from collections import deque
import math
import json
class Strategy:
    def __init__(self, symbol: Symbol, limit : int):
        self.symbol = symbol
        self.limit = limit
    def get_zscore(self, x: float, mean: float, std_dev: float) -> float:
        if std_dev == 0:
            return 0
        return (x - mean) / std_dev

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 0
        best_buy = max(order_depth.buy_orders.keys())
        best_sell = min(order_depth.sell_orders.keys())
        return (best_buy + best_sell) / 2
    
    def buy(self, price: int, quantity: int):
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price: int, quantity: int):
        self.orders.append(Order(self.symbol, price, -quantity))

class BaseStrategy:
    def __init__(self, symbol: Symbol, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.last_price = None
        self.orders = []      
        self.conversions = 0        

        self.price = deque(maxlen=20)

    def convert(self, qty: int) -> None:
        self.conversions += qty

    def compute_mean_std(self, values: deque) -> tuple[float, float]:
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return mean, variance**0.5

    def run(self, state: TradingState) -> tuple[List[Order], int]:
        self.orders = []
        self.conversions = 0
        self.act(state)

        return self.orders, self.conversions

class SquidInkIndicatorStrategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit
        self.prices = deque(maxlen=20)
        self.rsi_prices = deque(maxlen=14)

    def update_indicators(self, price: float):
        self.prices.append(price)
        self.rsi_prices.append(price)

        if len(self.prices) < 20 or len(self.rsi_prices) < 14:
            return None, None, None, None

        ma = sum(self.prices) / len(self.prices)
        std = (sum((p - ma) ** 2 for p in self.prices) / len(self.prices)) ** 0.5
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std

        deltas = [self.rsi_prices[i+1] - self.rsi_prices[i] for i in range(len(self.rsi_prices)-1)]
        gains = sum(max(d, 0) for d in deltas) / 14
        losses = sum(-min(d, 0) for d in deltas) / 14
        rs = gains / losses if losses != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        return ma, upper_band, lower_band, rsi

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        product = self.product
        orders = {}

        if product not in state.order_depths:
            return {}

        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {}

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        ma, upper, lower, rsi = self.update_indicators(mid_price)
        if ma is None:
            return {}

        position = state.position.get(product, 0)
        orders[product] = []

        if mid_price < lower and rsi < 35 and position < self.limit:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            orders[product].append(Order(product, best_ask + 1 , qty))

        elif mid_price > upper and rsi > 65 and position > -self.limit:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            orders[product].append(Order(product, best_bid - 1, -qty))

        elif lower < mid_price < upper and abs(position) > 0:
            if position > 0:
                qty = min(position, order_depth.buy_orders[best_bid])
                orders[product].append(Order(product, best_bid - 1, -qty))
            elif position < 0:
                qty = min(-position, order_depth.sell_orders[best_ask])
                orders[product].append(Order(product, best_ask + 1, qty))

        return orders  

class ResinStrategy:

    def run(self, state: TradingState, limit: int):
        product = "RAINFOREST_RESIN"
        orders = {}
        conversions = 0

        order_depth = state.order_depths.get(product)
        if not order_depth or not order_depth.buy_orders or not order_depth.sell_orders:
            return {}, 0

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        volume_bid = order_depth.buy_orders[best_bid]
        volume_ask = order_depth.sell_orders[best_ask]

        mid_price = (best_bid + best_ask) / 2
        position = state.position.get(product, 0)
        product_orders = []
        SELL_THRESHOLD = 10003
        BUY_THRESHOLD = 99997

        if mid_price >= SELL_THRESHOLD:
            max_sell = min(limit + position, volume_bid) 
            product_orders.append(Order(product, best_bid, -max_sell))
            conversions += max_sell

        elif mid_price <= BUY_THRESHOLD:
            max_buy = min(limit - position, volume_ask) 
            product_orders.append(Order(product, best_ask, max_buy))
            conversions += max_buy

        if product_orders:
            orders[product] = product_orders

        return orders, conversions

class KelpStrategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit
        self.alpha = 0.1  
        self.margin = 3  
        self.fair_price = None

    def run(self, state: TradingState) -> List[Order]:
        order_depth = state.order_depths[self.product]
        orders = {}
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        if self.fair_price is None:
            self.fair_price = mid_price
        else:
            self.fair_price = self.alpha * mid_price + (1 - self.alpha) * self.fair_price

        position = state.position.get(self.product, 0)
        offset = max(1, (best_ask - best_bid) // 2)
        product_orders = []

        if mid_price < self.fair_price - self.margin and position < self.limit:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            product_orders.append(Order(self.product, best_ask + offset, qty))

        elif mid_price > self.fair_price + self.margin and position > -self.limit:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            product_orders.append(Order(self.product, best_bid - offset, -qty))
        if product_orders:
            orders[self.product] = product_orders

        return orders



class MomentumShortCroissants(BaseStrategy):
    def run(self, state: TradingState) -> List[Order]:
        result = []
        if self.symbol not in state.order_depths:
            return result

        order_depth = state.order_depths[self.symbol]
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            if self.last_price is not None and mid_price < self.last_price:
                quantity = min(order_depth.buy_orders[best_bid], self.limit)
                result.append(Order(self.symbol, best_bid, -quantity))
            elif self.last_price is not None and mid_price > self.last_price:
                quantity = min(order_depth.sell_orders[best_ask], self.limit)
                result.append(Order(self.symbol, best_ask, quantity))

            self.last_price = mid_price
        return result

class SquidInkStrategy:

    def __init__(self, product : Product, limit: int):
        self.recent_prices = deque(maxlen=50)
        self.strategy = Strategy()
        self.anchor_price = 2000
        self.product = product
        self.limit = limit

    def run(self, state: TradingState, full_state: TradingState, entry_prices: dict, last_trade_time: dict) -> Dict[str, List[Order]]:
        orders = {}

        if self.product not in state.order_depths:
            return {}

        order_depth = state.order_depths[self.product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {}

        timestamp = full_state.timestamp
        mid_price = self.strategy.get_mid_price(order_depth)
        self.recent_prices.append(mid_price)

        if len(self.recent_prices) < 50:
            return {}  
        
        historical_mean = 2034
        historical_std = 10.85
        rolling_mean = sum(self.recent_prices ) / len(self.recent_prices)
        rolling_std = (sum((p - rolling_mean) ** 2 for p in self.recent_prices) / len(self.recent_prices)) ** 0.5
        
        blend = 0.5
        
        blended_mean = blend * rolling_mean + (1 - blend) * historical_mean
        blended_std = blend * rolling_std + (1 - blend) * historical_std

        z = self.strategy.get_zscore(mid_price, blended_mean, blended_std)

        position = state.position.get(self.product, 0)
        orders[self.product] = []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        volume_bid = order_depth.buy_orders[best_bid]
        volume_ask = order_depth.sell_orders[best_ask]
        offset = max(1, (best_ask - best_bid) // 2)

        if z > 1.5:
            fraction = min(1.0, abs(z) / 2.5)
            sell_qty = int(min(self.limit + position, volume_bid) * fraction)
            if sell_qty > 0:
                entry_prices[self.product] = best_bid
                last_trade_time[self.product] = timestamp
                orders[self.product].append(Order(self.product, best_bid - offset, -sell_qty))

        elif z < -1.5:
            fraction = min(1.0, abs(z) / 2.5)
            buy_qty = int(min(self.limit - position, volume_ask) * fraction)
            if buy_qty > 0:
                entry_prices[self.product] = best_ask
                last_trade_time[self.product] = timestamp
                orders[self.product].append(Order(self.product, best_ask + offset, buy_qty))

        elif -0.5 < z < 0.5 and self.product in entry_prices:
            if position > 0 and mid_price >= entry_prices[self.product] + offset:
                exit_qty = min(position, volume_bid)
                if exit_qty > 0:
                    orders[self.product].append(Order(self.product, best_bid, -exit_qty))
            elif position < 0 and mid_price <= entry_prices[self.product] - offset:
                exit_qty = min(-position, volume_ask)
                if exit_qty > 0:
                    orders[self.product].append(Order(self.product, best_ask, exit_qty))

        return orders

class SquidInkIndicatorStrategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit
        self.prices = deque(maxlen=20)
        self.rsi_prices = deque(maxlen=14)

    def update_indicators(self, price: float):
        self.prices.append(price)
        self.rsi_prices.append(price)

        if len(self.prices) < 20 or len(self.rsi_prices) < 14:
            return None, None, None, None

        ma = sum(self.prices) / len(self.prices)
        std = (sum((p - ma) ** 2 for p in self.prices) / len(self.prices)) ** 0.5
        upper_band = ma + 2 * std
        lower_band = ma - 2 * std

        deltas = [self.rsi_prices[i+1] - self.rsi_prices[i] for i in range(len(self.rsi_prices)-1)]
        gains = sum(max(d, 0) for d in deltas) / 14
        losses = sum(-min(d, 0) for d in deltas) / 14
        rs = gains / losses if losses != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        return ma, upper_band, lower_band, rsi

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        product = self.product
        orders = {}

        if product not in state.order_depths:
            return {}

        order_depth = state.order_depths[product]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return {}

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        ma, upper, lower, rsi = self.update_indicators(mid_price)
        if ma is None:
            return {}

        position = state.position.get(product, 0)
        orders[product] = []

        if mid_price < lower and rsi < 35 and position < self.limit:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            orders[product].append(Order(product, best_ask, qty))

        elif mid_price > upper and rsi > 65 and position > -self.limit:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            orders[product].append(Order(product, best_bid, -qty))

        elif lower < mid_price < upper and abs(position) > 0:
            if position > 0:
                qty = min(position, order_depth.buy_orders[best_bid])
                orders[product].append(Order(product, best_bid, -qty))
            elif position < 0:
                qty = min(-position, order_depth.sell_orders[best_ask])
                orders[product].append(Order(product, best_ask, qty))

        return orders

class ResinStrategy:
    def __init__(self, symbol: Symbol = "RAINFOREST_RESIN", limit: int = 50):
        self.symbol = symbol
        self.limit = limit

    def run(self, state: TradingState) -> tuple[List[Order], int]:
        orders = []
        conversions = 0

        if self.symbol not in state.order_depths:
            return [], 0

        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return [], 0

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        volume_bid = order_depth.buy_orders[best_bid]
        volume_ask = order_depth.sell_orders[best_ask]

        mid_price = (best_bid + best_ask) / 2
        position = state.position.get(self.symbol, 0)

        SELL_THRESHOLD = 10003
        BUY_THRESHOLD = 99997

        if mid_price >= SELL_THRESHOLD:
            max_sell = min(self.limit + position, volume_bid)
            if max_sell > 0:
                orders.append(Order(self.symbol, best_bid, -max_sell))
                conversions += max_sell

        elif mid_price <= BUY_THRESHOLD:
            max_buy = min(self.limit - position, volume_ask)
            if max_buy > 0:
                orders.append(Order(self.symbol, best_ask, max_buy))
                conversions += max_buy

        return orders, conversions

class KelpStrategy:
    def __init__(self, product: str, limit: int):
        self.product = product
        self.limit = limit
        self.alpha = 0.1 
        self.margin = 3 
        self.fair_price = None

    def run(self, state: TradingState) -> List[Order]:
        order_depth = state.order_depths[self.product]
        orders = {}
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return []

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2

        if self.fair_price is None:
            self.fair_price = mid_price
        else:
            self.fair_price = self.alpha * mid_price + (1 - self.alpha) * self.fair_price

        position = state.position.get(self.product, 0)
        offset = max(1, (best_ask - best_bid) // 2)
        product_orders = []

        if mid_price < self.fair_price - self.margin and position < self.limit:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            product_orders.append(Order(self.product, best_ask + offset, qty))

        elif mid_price > self.fair_price + self.margin and position > -self.limit:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            product_orders.append(Order(self.product, best_bid - offset, -qty))
        if product_orders:
            orders[self.product] = product_orders

        return orders

class MomentumShortCroissants(BaseStrategy):
    def run(self, state: TradingState) -> List[Order]:
        orders = []
        if self.symbol not in state.order_depths:
            return orders

        order_depth = state.order_depths[self.symbol]
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2

            if self.last_price is not None:
                delta = mid_price - self.last_price
                if delta < -0.5:
                    quantity = min(order_depth.buy_orders[best_bid], self.limit)
                    orders.append(Order(self.symbol, best_bid, -quantity))
                elif delta > 0.5:
                    quantity = min(order_depth.sell_orders[best_ask], self.limit)
                    orders.append(Order(self.symbol, best_ask, quantity))

            self.last_price = mid_price

        return orders

class MeanReversionJams(BaseStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=20)

    def run(self, state: TradingState) -> List[Order]:
        orders = []
        if self.symbol not in state.order_depths:
            return orders

        order_depth = state.order_depths[self.symbol]
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            mid_price = (best_bid + best_ask) / 2

            self.prices.append(mid_price)
            if len(self.prices) < 3:
                return orders 

            mean, stddev = self.compute_mean_std(self.prices)
            z_score = (mid_price - mean) / stddev if stddev > 0 else 0

            position = state.position.get(self.symbol, 0)

            if z_score < -0.5 and position < self.limit:
                quantity = min(self.limit - position, order_depth.sell_orders[best_ask])
                orders.append(Order(self.symbol, best_ask, quantity))
            elif z_score > 0.5 and position > -self.limit:
                quantity = min(self.limit + position, order_depth.buy_orders[best_bid])
                orders.append(Order(self.symbol, best_bid, -quantity))
            elif abs(z_score) < 0.2 and abs(position) > 0:
                if position > 0:
                    orders.append(Order(self.symbol, best_bid, -position))
                else:
                    orders.append(Order(self.symbol, best_ask, -position))

        return orders


class MeanReversionDjembe(BaseStrategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=20)

    def run(self, state: TradingState) -> List[Order]:
        result = []
        if self.symbol not in state.order_depths:
            return result

        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        self.prices.append(mid_price)

        if len(self.prices) < 3:
            return result

        mean, std = self.compute_mean_std(self.prices)
        z = (mid_price - mean) / std if std > 0 else 0
        position = state.position.get(self.symbol, 0)

        if z < -0.8 and position < self.limit:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            result.append(Order(self.symbol, best_ask + 1, qty))
        elif z > 0.8 and position > -self.limit:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            result.append(Order(self.symbol, best_bid - 1, -qty))
        elif abs(z) < 0.3 and abs(position) > 0:
            if position > 0:
                result.append(Order(self.symbol, best_bid, -position))
            elif position < 0:
                result.append(Order(self.symbol, best_ask, -position))

        return result


class ShortPicnicBasket(BaseStrategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.upper_threshold = 58700
        self.lower_threshold = 58200  
        self.last_mid_prices = deque(maxlen=10)

    def run(self, state: TradingState) -> List[Order]:
        result = []
        if self.symbol not in state.order_depths:
            return result

        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        self.last_mid_prices.append(mid_price)

        position = state.position.get(self.symbol, 0)

        if mid_price > self.upper_threshold and position > -self.limit:
            fraction = min(1.0, (mid_price - self.upper_threshold) / 500.0)
            qty = int(min(self.limit + position, order_depth.buy_orders[best_bid]) * fraction)
            if qty > 0:
                result.append(Order(self.symbol, best_bid, -qty))

        elif mid_price < self.lower_threshold and position < 0:
            result.append(Order(self.symbol, best_ask, -position))

        return result

class ShortPicnicBasket2(BaseStrategy):
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.last_price = None
        self.high_threshold = 58900 
        self.low_threshold = 58000

    def run(self, state: TradingState) -> List[Order]:
        result = []
        if self.symbol not in state.order_depths:
            return result

        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return result

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        position = state.position.get(self.symbol, 0)

        if mid_price > self.high_threshold and position > -self.limit:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            result.append(Order(self.symbol, best_bid, -qty))

        elif mid_price < self.low_threshold and position < self.limit:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            result.append(Order(self.symbol, best_ask, qty))

        elif self.low_threshold + 200 < mid_price < self.high_threshold - 200 and abs(position) > 0:
            if position > 0:
                result.append(Order(self.symbol, best_bid, -position))
            elif position < 0:
                result.append(Order(self.symbol, best_ask, -position))

        self.last_price = mid_price
        return result


class VolcanicRockStrategy(BaseStrategy):
    def __init__(self, symbol: str = "VOLCANIC_ROCK", limit: int = 400):
        super().__init__(symbol, limit)
        self.prices = deque(maxlen=50)

    def run(self, state: TradingState) -> List[Order]:
        orders = []
        if self.symbol not in state.order_depths:
            return orders

        order_depth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid = (best_bid + best_ask) / 2
        self.prices.append(mid)

        if len(self.prices) < 20:
            return orders

        mean, std = self.compute_mean_std(self.prices)
        z = (mid - mean) / std if std > 0 else 0

        position = state.position.get(self.symbol, 0)
        margin = max(1, (best_ask - best_bid) // 2)

        vote_long = vote_short = 0
        for s in state.order_depths:
            if s.startswith("VOLCANIC_ROCK_VOUCHER"):
                vote_long += 1 
                vote_short += 1  

        if z < -1.5 or vote_long > vote_short + 2:
            qty = min(self.limit - position, order_depth.sell_orders[best_ask])
            orders.append(Order(self.symbol, best_ask + margin, qty))
        elif z > 1.5 or vote_short > vote_long + 2:
            qty = min(self.limit + position, order_depth.buy_orders[best_bid])
            orders.append(Order(self.symbol, best_bid - margin, -qty))

        return orders

class Signal(IntEnum):
    NEUTRAL = 0
    SHORT = 1
    LONG = 2

class SignalStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int):
        super().__init__(symbol, limit)
        self.signal = Signal.NEUTRAL

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if self.signal == Signal.NEUTRAL:
            if position < 0:
                self.buy(self.get_buy_price(order_depth), -position)
            elif position > 0:
                self.sell(self.get_sell_price(order_depth), position)
        elif self.signal == Signal.SHORT:
            self.sell(self.get_sell_price(order_depth), self.limit + position)
        elif self.signal == Signal.LONG:
            self.buy(self.get_buy_price(order_depth), self.limit - position)

    def get_buy_price(self, order_depth: OrderDepth) -> int:
        return min(order_depth.sell_orders)

    def get_sell_price(self, order_depth: OrderDepth) -> int:
        return max(order_depth.buy_orders)

    def save(self):
        return {"signal": self.signal.value}

    def load(self, data):
        self.signal = Signal(data.get("signal", 0))
    
    def run(self, state: TradingState) -> tuple[List[Order], int]:
        self.orders = []
        self.act(state)
        return self.orders, 0  

class VolcanicVoucherStrategy(SignalStrategy):
    def __init__(self, symbol: Symbol, strike_price: float, expiry_days: int, limit: int, iv_fit_coeffs=None):
        super().__init__(symbol, limit)
        self.strike_price = strike_price
        self.expiry_days = expiry_days
        self.cdf = NormalDist().cdf
        self.iv_fit_coeffs = iv_fit_coeffs or (0.0, 0.0, 0.22)
        self.max_fraction = 0.5 
        self.min_T_days = 2 

    def fitted_iv(self, m_t: float) -> float:
        a, b, c = self.iv_fit_coeffs
        return max(0.05, a * m_t ** 2 + b * m_t + c)

    def get_signal(self, state: TradingState) -> Signal | None:
        if "VOLCANIC_ROCK" not in state.order_depths or self.symbol not in state.order_depths:
            return None

        rock_state = state.order_depths["VOLCANIC_ROCK"]
        voucher_state = state.order_depths[self.symbol]
        rock_mid = self.get_mid_price(rock_state)
        voucher_mid = self.get_mid_price(voucher_state)

        if rock_mid is None or voucher_mid is None or rock_mid <= 0:
            return None

        T = self.expiry_days / 365
        if self.expiry_days < self.min_T_days:
            return Signal.NEUTRAL  

        m_t = math.log(self.strike_price / rock_mid) / math.sqrt(T)
        sigma = self.fitted_iv(m_t)
        bs_price = self.black_scholes(rock_mid, self.strike_price, T, 0, sigma)
        epsilon = max(3, 0.5 * sigma * math.sqrt(T))

        self.fair_value = bs_price
        self.diff = voucher_mid - bs_price
        self.epsilon = epsilon

        if self.diff > epsilon:
            return Signal.SHORT
        elif self.diff < -epsilon:
            return Signal.LONG
        return Signal.NEUTRAL

    def act(self, state: TradingState) -> None:
        new_signal = self.get_signal(state)
        if new_signal is not None:
            self.signal = new_signal

        position = state.position.get(self.symbol, 0)
        order_depth = state.order_depths[self.symbol]

        if abs(self.expiry_days) < self.min_T_days:
            return  

        if self.signal == Signal.NEUTRAL and position != 0:
            if position > 0:
                self.sell(self.get_sell_price(order_depth), position)
            else:
                self.buy(self.get_buy_price(order_depth), -position)

        elif self.signal == Signal.LONG and position < self.limit:
            scale = min(self.max_fraction, abs(self.diff) / self.epsilon)
            qty = int(min(self.limit - position, scale * order_depth.sell_orders[self.get_buy_price(order_depth)]))
            if qty > 0:
                self.buy(self.get_buy_price(order_depth), qty)

        elif self.signal == Signal.SHORT and position > -self.limit:
            scale = min(self.max_fraction, abs(self.diff) / self.epsilon)
            qty = int(min(self.limit + position, scale * order_depth.buy_orders[self.get_sell_price(order_depth)]))
            if qty > 0:
                self.sell(self.get_sell_price(order_depth), qty)

    def black_scholes(self, S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * self.cdf(d1) - K * math.exp(-r * T) * self.cdf(d2)

class MagnificentMacaronsStrategy(BaseStrategy):
    # This strategy did not do well
    def __init__(self, symbol, limit, conv_limit=10):
        super().__init__(symbol, limit)
        self.sun_buy = 30
        self.sun_sell = 70
        self.imp_tariff_max = 10
        self.exp_tariff_max = 10
        self.conv_limit = conv_limit
        self.conv_used = 0
        self.prices = deque(maxlen=20)
        self.ema = None
        self.ema_alpha = 0.1

    def run(self, state: TradingState):
        orders = []
        pos = state.position.get(self.symbol, 0)
        od = state.order_depths.get(self.symbol)
        if not od or not od.buy_orders or not od.sell_orders:
            return orders

        conv = state.observations.conversionObservations.get(self.symbol, None)
        if conv:
            exp_t = conv.exportTariff
            imp_t = conv.importTariff
            fee  = conv.transportFees
            sun  = conv.sunlightIndex
            sugar= conv.sugarPrice
            ask_c = conv.askPrice   
            bid_c = conv.bidPrice   
        else:
            exp_t = imp_t = fee = sun = sugar = ask_c = bid_c = 0

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mid = 0.5*(best_bid + best_ask)

        if conv and self.conv_used < self.conv_limit:
            cost_in = ask_c + fee + imp_t
            if best_bid > cost_in + 1:
                qty = min(self.conv_limit - self.conv_used, od.buy_orders[best_bid])
                self.conv_used += qty
                self.convert(qty)
            cost_out = bid_c - fee - exp_t
            if best_ask < cost_out - 1:
                qty = min(self.conv_limit - self.conv_used, od.sell_orders[best_ask])
                self.conv_used += qty
                self.convert(-qty)

        if sun > self.sun_sell and exp_t < self.exp_tariff_max and pos > -self.limit:
            q = min(self.limit + pos, od.buy_orders[best_bid])
            if q>0: orders.append(Order(self.symbol, best_bid, -q))

        elif sun < self.sun_buy and imp_t < self.imp_tariff_max and pos < self.limit:
            q = min(self.limit - pos, od.sell_orders[best_ask])
            if q>0: orders.append(Order(self.symbol, best_ask, q))

        self.prices.append(mid)
        if self.ema is None:
            self.ema = mid
        else:
            self.ema = self.ema_alpha*mid + (1-self.ema_alpha)*self.ema

        if len(self.prices) == self.prices.maxlen:
            avg = sum(self.prices)/len(self.prices)
            if pos>0 and mid < avg - 0.5:
                q = min(pos, od.buy_orders[best_bid])
                if q>0: orders.append(Order(self.symbol, best_bid, -q))
            if pos<0 and mid > avg + 0.5:
                q = min(-pos, od.sell_orders[best_ask])
                if q>0: orders.append(Order(self.symbol, best_ask, q))

        return orders
    
class MacaronConversionStrategy:
    # This strategy did not do well
    def __init__(self, symbol: Symbol, limit: int):
        self.symbol = symbol
        self.limit = limit
        self.base_epsilon = 1.0  

    def run(self, state: TradingState) -> tuple[list[Order], int]:
        orders: list[Order] = []
        conversions = 0

        od = state.order_depths.get(self.symbol)
        obs: ConversionObservation = state.observations.conversionObservations.get(self.symbol)
        if od is None or obs is None or not od.buy_orders or not od.sell_orders:
            return orders, conversions

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        mac_mid = (best_bid + best_ask) / 2

        sugar_od = state.order_depths.get(obs.sugarPrice is None and None or "SUGAR")
        if sugar_od is None or not sugar_od.buy_orders or not sugar_od.sell_orders:
            return orders, conversions
        sugar_mid = (max(sugar_od.buy_orders) + min(sugar_od.sell_orders)) / 2

        used = state.conversionLevels.get(self.symbol, 0)
        limit = state.conversionLimits.get(self.symbol, 10)
        remaining = max(0, limit - used)
        if remaining == 0:
            return orders, conversions

        T = 1.0 
        solar_adj = 1 + (obs.sunlightIndex - 50) / 100 
        eps = max(0.5, self.base_epsilon * solar_adj)

        cost_to_buy = obs.askPrice * sugar_mid + obs.transportFees + obs.importTariff
        proceeds_on_sell = obs.bidPrice * sugar_mid - obs.exportTariff

        if mac_mid > cost_to_buy + eps:
            qty = min(remaining, self.limit)
            orders.append(Order(self.symbol, math.floor(mac_mid), qty))
            conversions += qty

        elif proceeds_on_sell > mac_mid + eps:
            qty = min(remaining, self.limit)
            orders.append(Order(self.symbol, math.ceil(mac_mid), -qty))
            conversions += qty

        return orders, conversions

class Trader:
    def __init__(self):
        self.voucher_strikes = {
            "VOLCANIC_ROCK_VOUCHER_9500": 9500,
            "VOLCANIC_ROCK_VOUCHER_9750": 9750,
            "VOLCANIC_ROCK_VOUCHER_10000": 10000,
            "VOLCANIC_ROCK_VOUCHER_10250": 10250,
            "VOLCANIC_ROCK_VOUCHER_10500": 10500,
        }
        self.voucher_limits = 200
        self.max_days = 7
        self.entry_prices = {}
        self.last_trade_time = {}
        self.strategies = {
            "CROISSANT": MomentumShortCroissants("CROISSANT", 250),
            "JAM": MeanReversionJams("JAM", 350),
            "DJEMBE": MeanReversionDjembe("DJEMBE", 60),
            "PICNIC_BASKET1": ShortPicnicBasket("PICNIC_BASKET1", 60),
            "PICNIC_BASKET2": ShortPicnicBasket2("PICNIC_BASKET2", 100),
            "SQUID_INK": SquidInkIndicatorStrategy("SQUID_INK", 50),
            "KELP": KelpStrategy("KELP", 50),
            "RAINFOREST_RESIN": ResinStrategy("RAINFOREST_RESIN", 50),
            "VOLCANIC_ROCK": VolcanicRockStrategy("VOLCANIC_ROCK", 400),
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy("MAGNIFICENT_MACARONS", 75)
        }
        self.macaron_conv = MacaronConversionStrategy("MAGNIFICENT_MACARONS", limit=10)
    def run(self, state: TradingState) -> tuple[Dict[Symbol, List[Order]], int, str]:
        orders: Dict[Symbol, List[Order]] = {}
        conversions = 0

        day_number = state.timestamp // 100_000
        expiry_days = max(1, self.max_days - day_number)

        old_data = json.loads(state.traderData) if state.traderData else {}
        new_data = {}

        for symbol, strike in self.voucher_strikes.items():
            strat = VolcanicVoucherStrategy(symbol, strike, expiry_days, self.voucher_limits)
            strat.orders = []
            if symbol in old_data:
                strat.load(old_data[symbol])
            self.strategies[symbol] = strat
        for symbol, strategy in self.strategies.items():
            if symbol not in state.order_depths or not state.order_depths[symbol].buy_orders or not state.order_depths[symbol].sell_orders:
                continue

            strat_orders = []
            strategy_result = strategy.run(state)
            conv_orders, strat_convs = self.macaron_conv.run(state)
            for o in conv_orders:
                orders.setdefault(o.symbol, []).append(o)
            conversions += strat_convs

            if isinstance(strategy_result, tuple):
                strat_orders, strat_conversions = strategy_result
                conversions += strat_conversions
            elif isinstance(strategy_result, list):
                strat_orders = strategy_result
            elif isinstance(strategy_result, dict):
                for sym, order_list in strategy_result.items():
                    orders.setdefault(sym, []).extend(order_list)
                continue  

            for order in strat_orders:
                orders.setdefault(order.symbol, []).append(order)

            if hasattr(strategy, "save"):
                new_data[symbol] = strategy.save()

        return orders, conversions, json.dumps(new_data, separators=(",", ":"))

