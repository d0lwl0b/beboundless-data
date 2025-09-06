# basescan.py
from dotenv import load_dotenv
load_dotenv()

import os
import random
from time import sleep
from pathlib import Path

class LogPrint():
    @staticmethod
    def info(msg):
        print(f"[INFO]::{msg}")
    @staticmethod
    def warning(msg):
        print(f"[WARNING]::{msg}")
    @staticmethod
    def error(msg):
        print(f"[ERROR]::{msg}")

# Request
import pandas as pd
from dataclasses import dataclass, asdict

from statistics import mean
from scipy.stats import trim_mean, median_abs_deviation, iqr
from botasaurus.request import request, Request, NotFoundException


@dataclass
class TxListRequest:
   address: str
   offset: int = 300

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY") or os.getenv("BASESCAN_API_KEY")

MONITOR_ADDRESSES = [
    "0x58db69748e3597e7e1ec55478a6edb92374169f0"
]

@request(cache=False, max_retry=10, retry_wait=1, raise_exception=True, create_error_logs=True)
def fetch_latest_gas_price(request: Request, params: TxListRequest):
    sleep(random.uniform(0.1, 0.3))  # To avoid rate limiting
    address = params.address
    offset = min(max(10, params.offset), 5000)
    url = (
        "https://api.etherscan.io/v2/api"
        f"?chainid=8453"
        f"&module=account"
        f"&action=txlist"
        f"&address={address}"
        f"&page=1"
        f"&offset={offset}"
        f"&sort=desc"
        f"&apikey={ETHERSCAN_API_KEY}"
    )
    response = request.get(url)
    if response.status_code == 404:
        raise NotFoundException(address)
    response.raise_for_status()
    data = response.json()
    if data.get("status") == "1" and data.get("result"):
        result_gen = (
            {
                "is_error": bool(int(tx.get("isError"))),
                "time_stamp": int(tx.get("timeStamp")),
                "gas_prices": int(tx.get("gasPrice"))
            } 
            for tx in data["result"]
            if tx.get("methodId") == "0xb4206dd2" and tx.get("isError") and tx.get("gasPrice") and tx.get("timeStamp")
        )

        result = []
        error_count = 0
        for item in result_gen:
            result.append(item)
            if item["is_error"]:
                error_count += 1

        result_length = len(result)
        error_rate = error_count / result_length if result_length else 0

        LogPrint.info(f"[{address}] [tx: {result_length / offset :.2%}] [error: {error_rate:.2%}] ({error_count} | {result_length} | {offset})")

        return result, error_rate
    return [], None


# Analysis
from enum import Enum
import numpy as np

class GasQuantile(Enum):
    LOW = 'low'
    MID = 'mid'
    HIGH = 'high'

class ExecMode(Enum):
    UP = 'up'
    CHASE = 'chase'
    RANDOM = 'random'

@dataclass
class BaseStats:
    min: float
    max: float
    mean: float

@dataclass
class AnalyzeStats:
    low: BaseStats
    mid: BaseStats
    high: BaseStats
    s_time: int
    e_time: int
    latest3: pd.DataFrame
    latest3_tags: list[GasQuantile]

def anti_normal_gas_iter(min_gas, max_gas, n_samples, edge_width=0.13, mid_gas=None):
    time_points = np.linspace(0, 100, n_samples)
    uniform_samples = np.random.uniform(-1, 1, n_samples)
    transformed_samples = np.sign(uniform_samples) * (1 - np.power(1 - np.abs(uniform_samples), 1/edge_width))
    if mid_gas is None:
        mid_gas = (min_gas + max_gas) / 2
    half_range = (max_gas - min_gas) / 2
    min_gas_price = random.uniform(1e8, 3e8)  # 0.s1 Gwei
    for t, s in zip(time_points, transformed_samples):
        price = mid_gas + s * half_range
        yield t, max(price, min_gas_price)

def quantile_to_tuple(q):
    return (q == GasQuantile.LOW, q == GasQuantile.MID, q == GasQuantile.HIGH)

def tuple_to_bitmask(t):
    # low=1, mid=2, high=4
    return t[0]*1 + t[1]*2 + t[2]*4

def analyze_gas_prices(data):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(data)

    # Sort by gas_prices
    df_gas_sorted = df.sort_values('gas_prices')
    df_time_sorted = df.sort_values('time_stamp', ascending=False)

    # Segment into 3 quantile bins: 0-20%, 20-80%, 80-100%
    df_gas_sorted['quantile'] = pd.qcut(df_gas_sorted['gas_prices'], q=[0, 0.2, 0.8, 1.0], labels=[GasQuantile.LOW, GasQuantile.MID, GasQuantile.HIGH])
    stats = {}
    for label, key in [(GasQuantile.LOW, 'low'), (GasQuantile.MID, 'mid'), (GasQuantile.HIGH, 'high')]:
        segment = df_gas_sorted[df_gas_sorted['quantile'] == label]['gas_prices']
        stats[key] = BaseStats(
            min=segment.min(),
            max=segment.max(),
            mean=segment.mean()
        )

    def get_quantile(price):
        row = df_gas_sorted[df_gas_sorted['gas_prices'] == price]
        return row['quantile'].iloc[0] if not row.empty else None

    # time analysis
    s_time = int(df_time_sorted['time_stamp'].iloc[-1])  # least time
    e_time = int(df_time_sorted['time_stamp'].iloc[0])   # latest time

    latest3 = df_time_sorted.head(3)
    latest_gas_prices = latest3['gas_prices'].values
    latest_quantiles = [get_quantile(p) for p in latest_gas_prices]

    return AnalyzeStats(**stats, s_time=s_time, e_time=e_time, latest3=latest3, latest3_tags=latest_quantiles)

# Main
import argparse
from collections import deque
from tomlkit import parse, dumps, table

def update_toml_price(toml_path, price):
    # Ensure price is non-negative
    min_gas_price = random.uniform(1e8, 3e8)  # 0.1 Gwei
    safe_price = max(price, min_gas_price)
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = parse(f.read())
    if "flashblocks" not in doc:
        doc["flashblocks"] = table()
    doc["flashblocks"]["initial_max_priority_fee_per_gas_wei"] = int(safe_price)
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))
    LogPrint.info(f">> updated [{int(safe_price) / 1e9}] [{int(safe_price)}] {toml_path} [flashblocks].initial_max_priority_fee_per_gas_wei")

def main(loop=False, interval=60, toml_path=None, factor=1.07, max_gas=int(3e9), offset=300):
    mode = ExecMode.UP
    area_data = {}
    mine_area = GasQuantile.LOW

    gas_price = None
    last_gas_price = None
    history_gas_prices = deque(maxlen=7)

    while True:
        # get data
        try:
            results = fetch_latest_gas_price([TxListRequest(address=address, offset=offset) for address in MONITOR_ADDRESSES])
        except Exception as e:
            LogPrint.error(f"Exception for {MONITOR_ADDRESSES}: {e}")
            results = None
        if not results:
            sleep(interval / 2)
            continue

        # extract valid prices
        analyze_data = None
        for address, (tx_list, error_rate) in zip(MONITOR_ADDRESSES, results):
            analyze_data = analyze_gas_prices(tx_list)
            area_data[address] = {}
            samples = 9

            is_low_change_ratio = False
            is_high_change_ratio = False

            if history_gas_prices and history_gas_prices[-1][0] is not None:
                last_analyze_data = history_gas_prices[-1][0]
                low_change_ratio = abs(analyze_data.low.mean - last_analyze_data.low.mean) / last_analyze_data.low.mean
                high_change_ratio = abs(analyze_data.high.mean - last_analyze_data.high.mean) / last_analyze_data.high.mean
                if low_change_ratio > 0.17:
                    is_low_change_ratio = True
                if high_change_ratio > 0.17:
                    is_high_change_ratio = True
            else:
                is_low_change_ratio = True
                is_high_change_ratio = True

            if is_low_change_ratio or not area_data[address].get(GasQuantile.HIGH, None):
                area_data[address][GasQuantile.HIGH] = anti_normal_gas_iter(
                    min_gas=analyze_data.high.min,
                    max_gas=analyze_data.high.max,
                    n_samples=samples,
                    edge_width=0.23,
                    mid_gas=analyze_data.high.mean * factor
                )

            if is_high_change_ratio or not area_data[address].get(GasQuantile.LOW, None):
                area_data[address][GasQuantile.LOW] = anti_normal_gas_iter(
                    min_gas=analyze_data.low.min,
                    max_gas=analyze_data.low.max,
                    n_samples=samples,
                    edge_width=0.23,
                    mid_gas=analyze_data.low.mean * factor
                )

        # TODO: calculate gas price
        is_error_values = analyze_data.latest3['is_error'].values
        xor_result = is_error_values[0] ^ is_error_values[1] ^ is_error_values[2]
        score = sum([tuple_to_bitmask(quantile_to_tuple(q)) for q in analyze_data.latest3_tags])
        if xor_result:
            use_factor = factor * (factor + 1 / score)
        else:
            use_factor = factor

        error_rate = results[0][1]  # use the first address's error rate for decision
        num_error_rate = min(int(error_rate * 100 // 10) + 1, 9)

        if error_rate is not None and num_error_rate % 3 == 0:
            mode = ExecMode.CHASE

        match mode:
            case ExecMode.CHASE:
                chase_factor = factor + 1 / score if xor_result else factor
                gas_price = analyze_data.mid.mean * chase_factor * chase_factor
                gas_price = gas_price + (last_gas_price if last_gas_price is not None else 0) * random.uniform(0.68, 0.96)
                if gas_price >= max_gas * factor:
                    gas_price = analyze_data.low.max * (1 / use_factor)
                if toml_path:
                    update_toml_price(toml_path, gas_price)
                LogPrint.info(
                    f"[CHASE] [price: {gas_price / 1e9}] [error_rate: {error_rate:.2%}] [mode: {mode}]"
                    f" | mid: ({analyze_data.mid.min / 1e9:.1f}, {analyze_data.mid.mean / 1e9:.1f}, {analyze_data.mid.max / 1e9:.1f})"
                    f"\n>>> xor: {xor_result}"
                )
                last_gas_price = gas_price
                mode = ExecMode.UP
            case ExecMode.UP:
                if gas_price is None:
                    gas_price = analyze_data.low.mean * use_factor
                else:
                    gas_price = gas_price * use_factor
                if last_gas_price and last_gas_price >= max_gas:
                    mode = ExecMode.RANDOM
                    continue
                if toml_path:
                    update_toml_price(toml_path, gas_price)
                LogPrint.info(
                    f"[price: {gas_price / 1e9}] [mode: {mode}]"
                    f" | low: ({analyze_data.low.min / 1e9:.1f}, {analyze_data.low.mean / 1e9:.1f}, {analyze_data.low.max / 1e9:.1f})"
                    f" | mid: ({analyze_data.mid.min / 1e9:.1f}, {analyze_data.mid.mean / 1e9:.1f}, {analyze_data.mid.max / 1e9:.1f})"
                    f" | high: ({analyze_data.high.min / 1e9:.1f}, {analyze_data.high.mean / 1e9:.1f}, {analyze_data.high.max / 1e9:.1f})"
                    f"\n>>> xor: {xor_result}"
                )
                last_gas_price = gas_price
            case ExecMode.RANDOM:
                # score in [3~12]
                LogPrint.info(f"[RANDOM] score: {score} from latest3 tags: {[q.value if q else None for q in analyze_data.latest3_tags]}")
                match score:
                    case 3 | 4 | 5 | 6:  # low
                        mine_area = GasQuantile.HIGH
                    case 7 | 8:  # mid
                        mode = ExecMode.UP
                        gas_price = analyze_data.low.min
                        last_gas_price = gas_price
                        continue
                    case 9 | 10 | 11 | 12:  # high
                        mine_area = GasQuantile.LOW

                for _, price in area_data[MONITOR_ADDRESSES[0]][mine_area]:
                    base_factor = factor + 1 / score
                    if price >= max_gas * factor:
                        pay_price = price * (1 / base_factor)
                    else:
                        pay_price = price * base_factor
                    if toml_path:
                        update_toml_price(toml_path, pay_price)
                    LogPrint.info(
                        f"[price: {pay_price / 1e9}] [xor: {xor_result}] [mode: {mode}] [area: {mine_area}] [score: {score}]")
                    sleep(interval + random.uniform(-interval*0.2, interval*0.1))
                else:
                    mode = ExecMode.UP
                    if mine_area == GasQuantile.HIGH:
                        gas_price = analyze_data.low.min
                    if mine_area == GasQuantile.LOW:
                        gas_price = analyze_data.mid.mean
                    last_gas_price = gas_price
                    area_data[MONITOR_ADDRESSES[0]][mine_area] = None
                    continue

        history_gas_prices.append((analyze_data, gas_price))
        # END
        sleep(interval)
        if not loop:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loop", action="store_true", help="Enable periodic monitoring loop")
    parser.add_argument("-i", "--interval", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("-t", "--toml-path", type=str, required=True, help="Path to the toml file to update")
    parser.add_argument("-f", "--factor", type=float, default=1.07, help="Factor for price adjustment")
    parser.add_argument("-x", "--max-gas", type=int, default=int(3e9), help="Maximum allowed gas price")
    parser.add_argument("-o", "--offset", type=int, default=300, help="Number of transactions to fetch (default 300, max 5000)")
    args = parser.parse_args()
    main(
        loop=args.loop,
        interval=args.interval,
        toml_path=args.toml_path,
        factor=args.factor,
        max_gas=args.max_gas,
        offset=args.offset
    )
