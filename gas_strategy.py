# gas_strategy.py

import argparse
import time
from pathlib import Path
import numpy as np
from tomlkit import parse, dumps, table

class LogPrint:
    @staticmethod
    def info(msg: str):
        print(f"[INFO]::{msg}")
    @staticmethod
    def warning(msg: str):
        print(f"[WARNING]::{msg}")
    @staticmethod
    def error(msg: str):
        print(f"[ERROR]::{msg}")

def update_toml_price(toml_path: Path, price: int, mode: str):
    # Write price to TOML file by mode
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = parse(f.read())
    if mode == "flashblocks":
        if "flashblocks" not in doc:
            doc["flashblocks"] = table()
        doc["flashblocks"]["initial_max_priority_fee_per_gas_wei"] = int(price)
        LogPrint.info(f"Updated [flashblocks] price: {int(price)}")
    elif mode == "market":
        if "market" not in doc:
            doc["market"] = table()
        doc["market"]["lockin_priority_gas"] = int(price)
        LogPrint.info(f"Updated [market] price: {int(price)}")
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))

def strategy_loop(
    loop: bool,
    low_price: int,
    multiplier_min: float,
    multiplier_max: float,
    low_period: int,
    high_period: int,
    toml_path: str,
    mode: str
):
    rng = np.random.default_rng()
    while True:
        # Low price stage
        update_toml_price(Path(toml_path), low_price, mode)
        time.sleep(low_period)
        # High price stage
        multiplier = rng.uniform(multiplier_min, multiplier_max)
        high_price = int(low_price * multiplier)
        update_toml_price(Path(toml_path), high_price, mode)
        time.sleep(high_period)
        if not loop:
            break

def main():
    parser = argparse.ArgumentParser(description="Price cycle strategy")
    parser.add_argument("--loop", action="store_true", help="Enable loop")
    parser.add_argument("--low-price", type=int, required=True, help="Low price (wei)")
    parser.add_argument("--multiplier-min", type=float, default=1.2, help="Min multiplier")
    parser.add_argument("--multiplier-max", type=float, default=2.0, help="Max multiplier")
    parser.add_argument("--low-period", type=int, default=300, help="Low price duration (sec)")
    parser.add_argument("--high-period", type=int, default=10, help="High price duration (sec)")
    parser.add_argument("--toml-path", type=str, required=True, help="TOML file path")
    parser.add_argument("--mode", type=str, choices=["flashblocks", "market"], required=True, help="Write mode")
    args = parser.parse_args()
    strategy_loop(
        loop=args.loop,
        low_price=args.low_price,
        multiplier_min=args.multiplier_min,
        multiplier_max=args.multiplier_max,
        low_period=args.low_period,
        high_period=args.high_period,
        toml_path=args.toml_path,
        mode=args.mode
    )

if __name__ == "__main__":
    main()
