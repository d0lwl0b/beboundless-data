import argparse
import time
from pathlib import Path
import numpy as np
from tomlkit import parse, dumps, table
from collections import deque
from enum import Enum

class PriceState(Enum):
    HIGH_PRICE = "high"
    LOW_PRICE = "low"

class LogPrint:
    @staticmethod
    def info(period: str, period_time: float, price: int, write: str, trigger: float, ratio: float):
        print(f"[period: {period} {period_time:.2f}s] [price: {price}] [write: {write}] [trigger: {trigger:.2%}] [ratio: {ratio:.3f}]")
    @staticmethod
    def trigger_reason(reason: str):
        print(f"[trigger: {reason}]")
    @staticmethod
    def error(msg: str):
        print(f"[ERROR]::{msg}")

def update_toml_price(toml_path: Path, price: int, mode: str, state: PriceState, trigger_history: deque, last_price: int, write_buffer: dict, last_write_time: float):
    """Write price to TOML file if state or price changes significantly, using buffer."""
    last_state = trigger_history[-1][0] if trigger_history else PriceState.LOW_PRICE
    if state == last_state and abs(price - last_price) / last_price < 0.1:
        write_buffer["price"] = price
        return last_price, last_write_time, "skipped"
    current_time = time.time()
    if current_time - last_write_time < 10 and write_buffer.get("price") == price:
        write_buffer["price"] = price
        return last_price, last_write_time, "skipped"
    try:
        with open(toml_path, "r", encoding="utf-8") as f:
            doc = parse(f.read())
        if mode == "flashblocks":
            if "flashblocks" not in doc:
                doc["flashblocks"] = table()
            doc["flashblocks"]["initial_max_priority_fee_per_gas_wei"] = int(price)
        elif mode == "market":
            if "market" not in doc:
                doc["market"] = table()
            doc["market"]["lockin_priority_gas"] = int(price)
        with open(toml_path, "w", encoding="utf-8") as f:
            f.write(dumps(doc))
        elapsed = (current_time - time.time()) * -1000
        write_buffer["price"] = price
        return price, current_time, f"{elapsed:.2f}ms"
    except Exception as e:
        LogPrint.error(f"Failed to update TOML: {e}")
        return last_price, last_write_time, "skipped"

def strategy_loop(
    loop: bool,
    low_price: int,
    multiplier_min: float,
    multiplier_max: float,
    toml_path: str,
    mode: str,
    n_machines: int,
    reverse_time: bool = False
):
    rng = np.random.default_rng()
    # Natural constant e and ratio bounds (low/high)
    e = 2.718281828459045
    ratio_min = e * 0.70  # e - 30%
    ratio_max = e * 0.77  # e - 23%
    # Window time parameters (normal distribution, 2σ range)
    high_period_mean = 4.5  # High price mean (seconds)
    high_period_sigma = 0.25  # Standard deviation for high period
    low_period_sigma = 0.3  # Standard deviation for low period
    # Sliding window and trigger control
    maxlen = 50
    trigger_history = deque(maxlen=maxlen)  # Track last 50 cycles (~650s), stores (PriceState, price)
    target_trigger_prob = 0.3  # Target 30% high price probability
    trigger_threshold_low = 0.27  # Force trigger if below 27%
    trigger_threshold_high = 0.33  # Skip trigger if above 33%
    no_trigger_count = 0  # Track consecutive non-triggers
    max_no_trigger = 3  # Force trigger after 3 non-triggers
    last_price = low_price  # Track last written price
    write_buffer = {"price": low_price}  # Buffer for writes
    last_write_time = time.time()  # Track last write time

    while True:
        # Generate high period (normal distribution, 2σ range)
        high_time = rng.normal(high_period_mean, high_period_sigma)
        high_time = max(4.2, min(4.7, high_time))  # Cap within 2σ
        # Generate ratio and calculate low period (low/high ≈ e)
        ratio = rng.uniform(ratio_min, ratio_max)  # Ratio in [2.636, 3.153]
        low_period_mean = high_time * ratio  # Dynamic low period mean
        low_time = rng.normal(low_period_mean, low_period_sigma)
        low_time = max(low_period_mean - 2 * low_period_sigma, min(low_period_mean + 2 * low_period_sigma, low_time))  # Cap within 2σ

        if reverse_time:
            high_time, low_time = low_time, high_time

        # Determine price and state
        state = PriceState.LOW_PRICE
        current_price = int(low_price * (1 + rng.choice([-1, 1]) * rng.uniform(0.2, 0.5)))
        trigger_reason = None
        if trigger_history and trigger_history[-1][0] == PriceState.HIGH_PRICE:
            trigger_reason = "forced_low"
        elif len(trigger_history) < maxlen:
            trigger_reason = "random"
            if rng.random() < target_trigger_prob:
                state = PriceState.HIGH_PRICE
        else:
            current_prob = sum(1 if state == PriceState.HIGH_PRICE else 0 for state, _ in trigger_history) / len(trigger_history)
            if current_prob < trigger_threshold_low or no_trigger_count >= max_no_trigger:
                state = PriceState.HIGH_PRICE
                trigger_reason = "forced_high"
            elif current_prob <= trigger_threshold_high and rng.random() < target_trigger_prob:
                state = PriceState.HIGH_PRICE
                trigger_reason = "random"
            elif current_prob > trigger_threshold_high:
                trigger_reason = "skipped"

        if state == PriceState.HIGH_PRICE:
            multiplier = multiplier_min + (multiplier_max - multiplier_min) * rng.power(2.0)
            current_price = int(low_price * multiplier)
            trigger_history.append((PriceState.HIGH_PRICE, current_price))  # Record high price
            no_trigger_count = 0  # Reset counter
        else:
            trigger_history.append((PriceState.LOW_PRICE, current_price))  # Record low price
            no_trigger_count += 1  # Increment counter

        # Update TOML file
        last_price, last_write_time, write_status = update_toml_price(
            Path(toml_path), current_price, mode, state, trigger_history, last_price, write_buffer, last_write_time
        )

        # Log key data
        trigger_rate = sum(1 if state == PriceState.HIGH_PRICE else 0 for state, _ in trigger_history) / len(trigger_history) if trigger_history else 0
        LogPrint.info(
            period=state.value,
            period_time=high_time if state == PriceState.HIGH_PRICE else low_time,
            price=current_price,
            write=write_status,
            trigger=trigger_rate,
            ratio=low_time / high_time
        )
        if trigger_reason:
            LogPrint.trigger_reason(trigger_reason)

        time.sleep(high_time if state == PriceState.HIGH_PRICE else low_time)

        if not loop:
            break

def main():
    parser = argparse.ArgumentParser(description="Price cycle strategy")
    parser.add_argument("--loop", action="store_true", help="Enable loop")
    parser.add_argument("--low-price", type=int, default=12000000, help="Low price (wei)")
    parser.add_argument("--multiplier-min", type=float, default=5.0, help="Min multiplier")
    parser.add_argument("--multiplier-max", type=float, default=15.0, help="Max multiplier")
    parser.add_argument("--toml-path", type=str, required=True, help="TOML file path")
    parser.add_argument("--mode", type=str, choices=["flashblocks", "market"], required=True, help="Write mode")
    parser.add_argument("--n-machines", type=int, default=30, help="Number of mining machines")
    parser.add_argument("--reverse-time", action="store_true", help="Reverse high/low time duration")
    args = parser.parse_args()
    
    strategy_loop(
        loop=args.loop,
        low_price=args.low_price,
        multiplier_min=args.multiplier_min,
        multiplier_max=args.multiplier_max,
        toml_path=args.toml_path,
        mode=args.mode,
        n_machines=args.n_machines,
        reverse_time=args.reverse_time
    )

if __name__ == "__main__":
    main()
