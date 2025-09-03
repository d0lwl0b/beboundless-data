# gas_noise.py
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from tomlkit import parse, dumps


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


def update_toml_price(toml_path: Path, price: int):
    # Write price into TOML at [market].lockin_priority_gas
    with open(toml_path, "r", encoding="utf-8") as f:
        doc = parse(f.read())
    if "market" not in doc:
        from tomlkit import table
        doc["market"] = table()
    doc["market"]["lockin_priority_gas"] = int(price)
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(dumps(doc))
    LogPrint.info(f"Updated {toml_path} [market].lockin_priority_gas = {int(price)}")


# -------- Anti-normal core: two truncated edge Gaussians with rejection --------
def anti_normal_sample(
    rng: np.random.Generator,
    min_gas: int,
    max_gas: int,
    *,
    edge_width_frac: float = 0.22,  # edge band width as fraction of total range
    sigma_frac: float = 0.05,       # Gaussian sigma as fraction of total range
    choose_low_prob: float = 0.5,   # probability to sample the low edge band
    max_tries: int = 256,
) -> int:
    R = max_gas - min_gas
    if R <= 0:
        return int(min_gas)

    w = float(np.clip(edge_width_frac, 1e-6, 0.49))
    sigma = max(1e-9, sigma_frac * R)

    low_lo,  low_hi  = min_gas,         min_gas + w * R
    high_lo, high_hi = max_gas - w * R, max_gas

    if rng.random() < choose_low_prob:
        mu, lo, hi = (low_lo + low_hi) / 2.0, low_lo, low_hi
    else:
        mu, lo, hi = (high_lo + high_hi) / 2.0, high_lo, high_hi

    for _ in range(max_tries):
        v = rng.normal(mu, sigma)
        if lo <= v <= hi:
            return int(round(v))

    # Fallback: band midpoint (should rarely happen)
    return int(round(mu))


def adjust_noise_range(
    history: List[int],
    min_gas: int,
    max_gas: int,
    max_consumption: Optional[int] = None,
) -> Tuple[int, int]:
    # If avg exceeds max_consumption, softly reduce max_gas
    if not history or max_consumption is None:
        return min_gas, max_gas
    avg = float(np.mean(history))
    if avg > max_consumption:
        new_max = max(min_gas + 1, int(avg * 0.95))
        if new_max < max_gas:
            LogPrint.warning(
                f"Avg gas {avg:.2f} > max_consumption {max_consumption}, shrink max_gas to {new_max}"
            )
            return min_gas, new_max
    return min_gas, max_gas


def generate_gas_sequence(
    min_gas: int,
    max_gas: int,
    *,
    steps: int = 100,
    switch_period: int = 0,          # every N steps flip bias to the other edge
    interval_min: float = 1.0,
    interval_max: float = 2.0,
    max_consumption: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,  # reuse in loop mode
    seed: Optional[int] = None,                 # used only if rng is None
    toml_path: Optional[str] = None,
    write_toml: bool = False,
    verbose: bool = False,
    edge_width_frac: float = 0.22,
    sigma_frac: float = 0.05,
) -> Tuple[List[int], List[float]]:
    if min_gas >= max_gas:
        raise ValueError("min_gas must be < max_gas")
    if interval_min < 0 or interval_max < interval_min:
        raise ValueError("invalid interval range")

    rng = rng or np.random.default_rng(seed)
    history: List[int] = []
    prices: List[int] = []
    times: List[float] = []
    t0 = time.time()

    for i in range(int(steps)):
        min_gas, max_gas = adjust_noise_range(history, min_gas, max_gas, max_consumption)

        # Bias alternates by block when switch_period > 0; otherwise 50/50
        if switch_period and switch_period > 0:
            block = (i // switch_period) % 2
            choose_low_prob = 0.8 if block == 0 else 0.2
        else:
            choose_low_prob = 0.5

        price = anti_normal_sample(
            rng=rng,
            min_gas=min_gas,
            max_gas=max_gas,
            edge_width_frac=edge_width_frac,
            sigma_frac=sigma_frac,
            choose_low_prob=choose_low_prob,
        )

        if write_toml and toml_path:
            try:
                update_toml_price(Path(toml_path), price)
            except Exception as e:
                LogPrint.error(f"Failed to update TOML: {e}")

        history.append(price)
        if len(history) > 100:
            history = history[-100:]
        prices.append(price)
        times.append(time.time() - t0)

        dt = float(rng.uniform(interval_min, interval_max))
        if verbose:
            side = "LOW" if choose_low_prob >= 0.5 else "HIGH"
            LogPrint.info(f"price {price} (side={side}, edge_w={edge_width_frac}, sigma={sigma_frac})")
            LogPrint.info(f"sleep {dt:.2f}")
        time.sleep(dt)

    return prices, times


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Anti-normal gas sequence generator")
    p.add_argument("--min-gas", type=int, required=True, help="min gas (wei)")
    p.add_argument("--max-gas", type=int, required=True, help="max gas (wei)")
    p.add_argument("--interval", type=float, nargs=2, required=True, metavar=("MIN", "MAX"), help="seconds range")
    p.add_argument("--loop", action="store_true", help="loop mode: sample once per cycle")
    p.add_argument("--steps", type=int, default=100, help="steps in non-loop mode")
    p.add_argument("--switch-period", type=int, default=0, help="flip bias every N steps (0=off)")
    p.add_argument("--max-consumption", type=int, default=None, help="soft cap for recent average")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--toml-path", type=str, default=None, help="path to TOML for price updates")
    p.add_argument("--verbose", action="store_true", help="verbose logs")
    p.add_argument("--edge-width", type=float, default=0.22, help="edge band width fraction (0~0.49)")
    p.add_argument("--sigma-frac", type=float, default=0.05, help="edge Gaussian sigma fraction")
    return p.parse_args()


def main():
    args = parse_args()

    min_gas, max_gas = args.min_gas, args.max_gas
    interval_min, interval_max = args.interval
    toml_path = args.toml_path

    # One RNG for the whole run; reused in loop mode to avoid repeating sequences
    rng = np.random.default_rng(args.seed)

    if args.loop:
        LogPrint.info("Loop mode on. Press Ctrl+C to stop.")
        try:
            while True:
                generate_gas_sequence(
                    min_gas=min_gas,
                    max_gas=max_gas,
                    steps=1,
                    switch_period=args.switch_period,
                    interval_min=interval_min,
                    interval_max=interval_max,
                    max_consumption=args.max_consumption,
                    rng=rng,
                    toml_path=toml_path,
                    write_toml=bool(toml_path),
                    verbose=args.verbose,
                    edge_width_frac=args.edge_width,
                    sigma_frac=args.sigma_frac,
                )
        except KeyboardInterrupt:
            LogPrint.info("Stopped by user.")
    else:
        generate_gas_sequence(
            min_gas=min_gas,
            max_gas=max_gas,
            steps=int(args.steps),
            switch_period=args.switch_period,
            interval_min=interval_min,
            interval_max=interval_max,
            max_consumption=args.max_consumption,
            rng=rng,
            toml_path=toml_path,
            write_toml=bool(toml_path),
            verbose=args.verbose,
            edge_width_frac=args.edge_width,
            sigma_frac=args.sigma_frac,
        )


if __name__ == "__main__":
    main()
