# basescan.py
from dotenv import load_dotenv
load_dotenv()

import os
import random
from time import sleep
from pathlib import Path

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY") or os.getenv("BASESCAN_API_KEY")

MONITOR_ADDRESSES = [
    "0x58db69748e3597e7e1ec55478a6edb92374169f0",
]

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
from statistics import mean
from scipy.stats import trim_mean, median_abs_deviation, iqr
from botasaurus.request import request, Request, NotFoundException

@request(cache=False, max_retry=10, retry_wait=1, raise_exception=True, create_error_logs=True)
def fetch_latest_gas_price(request: Request, address: str, offset: int = 300) -> tuple[int | None, int | None, int | None, float]:
    """
    Returns (low_mean, mid_mean, high_mean, change_rate)
    - low_mean: mean of lowest 10%
    - mid_mean: mean of middle 80%
    - high_mean: mean of highest 10%
    - change_rate: normalized MAD / mid_mean
    - offset: number of transactions to fetch (default 300, max 5000)
    """
    sleep(random.uniform(0.1, 0.3))  # To avoid rate limiting
    offset = min(max(100, offset), 5000)
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
        txs = [tx for tx in data["result"] if tx.get("methodId") == "0xb4206dd2"]
        gas_prices = sorted([int(tx.get("gasPrice", "0")) for tx in txs if tx.get("gasPrice")])
        n = len(gas_prices)
        if n == 0:
            return None, None, None, 0.0
        low_count  = max(1, int(n * 0.1))
        high_count = max(1, int(n * 0.1))
        mid_count  = n - low_count - high_count

        low_slice  = gas_prices[:low_count]
        mid_slice  = gas_prices[low_count:n-high_count] if mid_count > 0 else []
        high_slice = gas_prices[n-high_count:]

        low_mean  = int(mean(low_slice))  if low_slice  else None
        mid_mean  = int(mean(mid_slice))  if mid_slice  else None
        high_mean = int(mean(high_slice)) if high_slice else None

        mad = median_abs_deviation(mid_slice, scale=1.0) if mid_slice else 0.0
        change_rate = float(min(1.0, mad / mid_mean)) if mid_mean else 0.0
        return low_mean, mid_mean, high_mean, change_rate
    return None, None, None, 0.0


# Main
import math
from tomlkit import parse, dumps
import argparse
import numpy as np  # For anti-normal sampling

# Anti-normal sampling: dense at edges, sparse in middle
def anti_normal_sample(
    min_gas: int,
    max_gas: int,
    edge_width_frac: float = 0.13,
    sigma_frac: float = 0.04,
    choose_low_prob: float = 0.5,
    max_tries: int = 32,
) -> int:
    R = max_gas - min_gas
    if R <= 0:
        return int(min_gas)
    w = float(np.clip(edge_width_frac, 1e-6, 0.49))
    sigma = max(1e-9, sigma_frac * R)
    low_lo, low_hi = min_gas, min_gas + w * R
    high_lo, high_hi = max_gas - w * R, max_gas
    if np.random.rand() < choose_low_prob:
        mu, lo, hi = (low_lo + low_hi) / 2.0, low_lo, low_hi
    else:
        mu, lo, hi = (high_lo + high_hi) / 2.0, high_lo, high_hi
    for _ in range(max_tries):
        v = np.random.normal(mu, sigma)
        if lo <= v <= hi:
            return int(round(v))
    return int(round(mu))

def update_toml_price(toml_path, price):
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = parse(f.read())
    if "flashblocks" not in doc:
        from tomlkit import table
        doc["flashblocks"] = table()
    doc["flashblocks"]["initial_max_priority_fee_per_gas_wei"] = int(price)
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))
    LogPrint.info(f"Updated {toml_path} [flashblocks].initial_max_priority_fee_per_gas_wei = {int(price)}")

def main(loop=False, interval=60, toml_path=None, min_factor=1.05, max_factor=1.20, max_gas=int(3e9), offset=300):
    """
    Main monitoring loop. Dynamically adjusts gas price using adaptive factor and max_gas.
    If price exceeds max_gas, enters cooldown period with minimum gas.
    offset: number of transactions to fetch (default 300, max 5000)
    """
    min_gas = int(1e9)  # Minimum gas price during cooldown
    while True:
        try:
            results = fetch_latest_gas_price(MONITOR_ADDRESSES, offset=offset)
        except Exception as e:
            LogPrint.error(f"Exception for {MONITOR_ADDRESSES}: {e}")
            results = [(None, None, None, 0.0)] * len(MONITOR_ADDRESSES)
        valid_prices = []
        for address, (low_mean, mid_mean, high_mean, change_rate) in zip(MONITOR_ADDRESSES, results):
            factor = min_factor + (max_factor - min_factor) * change_rate
            factor = max(factor, 1.0)
            LogPrint.info(
                f"{address} low_mean={low_mean} mid_mean={mid_mean} high_mean={high_mean} change_rate={change_rate:.3f} factor={factor:.3f}"
            )
            if mid_mean is not None:
                # Use anti-normal sampling between low_mean and high_mean
                min_sample = int(low_mean if low_mean is not None else mid_mean * 0.9)
                max_sample = int(high_mean if high_mean is not None else mid_mean * 1.1)
                sampled_price = anti_normal_sample(
                    min_gas=min_sample,
                    max_gas=max_sample,
                    edge_width_frac=0.22,
                    sigma_frac=0.05,
                    choose_low_prob=0.5,
                )
                LogPrint.info(
                    f"Anti-normal sample: [{min_sample}, {max_sample}], price={sampled_price}"
                )
                if sampled_price > max_gas or sampled_price < mid_mean:
                    min_gas = low_mean if low_mean else min_gas
                    update_toml_price(toml_path, int(min_gas // factor))
                    LogPrint.info(f"Cooldown: price {sampled_price} exceeds max_gas {max_gas}, set to low_mean {low_mean}")
                else:
                    update_toml_price(toml_path, int(sampled_price * factor))
                    LogPrint.info(f"Set price: {sampled_price} (anti-normal, max_gas={max_gas})")
                valid_prices.append(sampled_price)
            else:
                LogPrint.error(f"Failed to fetch gas price for {address}")
        if not loop:
            break
        sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loop", action="store_true", help="Enable periodic monitoring loop")
    parser.add_argument("-i", "--interval", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("-t", "--toml-path", type=str, required=True, help="Path to the toml file to update")
    parser.add_argument("--min-factor", type=float, default=1.05, help="Minimum factor for price adjustment")
    parser.add_argument("--max-factor", type=float, default=1.20, help="Maximum factor for price adjustment")
    parser.add_argument("-x", "--max-gas", type=int, default=int(3e9), help="Maximum allowed gas price")
    parser.add_argument("-o", "--offset", type=int, default=300, help="Number of transactions to fetch (default 300, max 5000)")
    args = parser.parse_args()
    main(
        loop=args.loop,
        interval=args.interval,
        toml_path=args.toml_path,
        min_factor=args.min_factor,
        max_factor=args.max_factor,
        max_gas=args.max_gas,
        offset=args.offset
    )
