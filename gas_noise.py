# gas_noise.py

import argparse
import random
import time
from pathlib import Path
from tomlkit import parse, dumps
import numpy as np

try:
    from noise import pnoise1
except ImportError:
    pnoise1 = None

class LogPrint:
    @staticmethod
    def info(msg):
        print(f"[INFO]::{msg}")
    @staticmethod
    def warning(msg):
        print(f"[WARNING]::{msg}")
    @staticmethod
    def error(msg):
        print(f"[ERROR]::{msg}")

def update_toml_price(toml_path, price):
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = parse(f.read())
    if "market" not in doc:
        from tomlkit import table
        doc["market"] = table()
    doc["market"]["lockin_priority_gas"] = int(price)
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))
    LogPrint.info(f"Updated {toml_path} [market].lockin_priority_gas = {int(price)}")

def perlin_noise_value(seed, min_gas, max_gas):
    if pnoise1 is None:
        raise ImportError("noise package not installed")
    v = pnoise1(seed)
    v = (v + 1) / 2  # [-1,1] -> [0,1]
    return int(min_gas + v * (max_gas - min_gas))

def gaussian_noise_value(rng, mean, std, min_gas, max_gas):
    v = rng.normal(mean, std)
    v = max(min_gas, min(max_gas, int(v)))
    return v

def adjust_noise_range(history, min_gas, max_gas, max_consumption=None):
    if not history or max_consumption is None:
        return min_gas, max_gas
    avg = np.mean(history)
    if avg > max_consumption:
        new_max = max(min_gas, int(avg * 0.95))
        LogPrint.warning(f"Avg gas {avg} > max_consumption {max_consumption}, shrink max_gas to {new_max}")
        return min_gas, new_max
    return min_gas, max_gas

def generate_gas_sequence(
    min_gas,
    max_gas,
    steps=100,
    switch_period=10,
    interval_min=1,
    interval_max=2,
    max_consumption=None,
    seed=None,
    toml_path=None,
    write_toml=False,
    verbose=False
):
    rng = np.random.default_rng(seed)
    history = []
    algo_flag = 0  # 0: perlin, 1: gaussian
    perlin_seed = rng.uniform(0, 1000)
    perlin_step = 0.1
    prices = []
    times = []
    t0 = time.time()
    for i in range(steps):
        min_gas, max_gas = adjust_noise_range(history, min_gas, max_gas, max_consumption)
        if algo_flag == 0:
            price = perlin_noise_value(perlin_seed, min_gas, max_gas)
            perlin_seed += perlin_step
            if verbose:
                print(f"perlin {price}")
        else:
            mean = (min_gas + max_gas) / 2
            std = (max_gas - min_gas) / 6 if max_gas > min_gas else 1
            price = gaussian_noise_value(rng, mean, std, min_gas, max_gas)
            if verbose:
                print(f"gauss {price}")
        if write_toml and toml_path:
            update_toml_price(toml_path, price)
        history.append(price)
        if len(history) > 100:
            history = history[-100:]
        prices.append(price)
        times.append(time.time() - t0)
        if (i + 1) % switch_period == 0:
            algo_flag = 1 - algo_flag
        sleep_time = rng.uniform(interval_min, interval_max)
        if verbose:
            print(f"sleep {sleep_time:.2f}")
        time.sleep(sleep_time)
    return prices, times

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--toml-path", type=str, required=False, help="Path to the toml file to update (optional)")
    parser.add_argument("--min-gas", type=int, required=True, help="Minimum gas price")
    parser.add_argument("--max-gas", type=int, required=True, help="Maximum gas price")
    parser.add_argument("--interval", type=int, nargs=2, required=True, metavar=("MIN", "MAX"), help="Interval range in seconds")
    parser.add_argument("--loop", action="store_true", help="Enable periodic loop")
    parser.add_argument("--switch-period", type=int, default=10, help="Algorithm switch period (times)")
    parser.add_argument("--max-consumption", type=int, default=None, help="Max allowed avg gas price (auto shrink range)")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run (default 100, ignored if --loop)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Print detailed info")
    return parser.parse_args()

def main():
    args = parse_args()
    min_gas, max_gas = args.min_gas, args.max_gas
    interval_min, interval_max = args.interval
    toml_path = args.toml_path
    switch_period = args.switch_period
    max_consumption = args.max_consumption
    seed = args.seed
    verbose = args.verbose

    if args.loop:
        while True:
            generate_gas_sequence(
                min_gas,
                max_gas,
                steps=1,
                switch_period=switch_period,
                interval_min=interval_min,
                interval_max=interval_max,
                max_consumption=max_consumption,
                seed=seed,
                toml_path=toml_path,
                write_toml=bool(toml_path),
                verbose=verbose
            )
    else:
        generate_gas_sequence(
            min_gas,
            max_gas,
            steps=args.steps,
            switch_period=switch_period,
            interval_min=interval_min,
            interval_max=interval_max,
            max_consumption=max_consumption,
            seed=seed,
            toml_path=toml_path,
            write_toml=bool(toml_path),
            verbose=verbose
        )

if __name__ == "__main__":
    main()
