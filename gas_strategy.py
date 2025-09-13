# gas_strategy.py

from sqlmodel import SQLModel, Field, Session, create_engine, select
from datetime import datetime, timedelta
from decimal import Decimal
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
    def info(msg: str):
        print(f"[INFO]::{msg}")

    @staticmethod
    def warning(msg: str):
        print(f"[WARNING]::{msg}")

    @staticmethod
    def error(msg: str):
        print(f"[ERROR]::{msg}")


class Measurement(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now, index=True)
    duration: timedelta = Field(default=timedelta(seconds=0))
    price: Decimal = Field(default=Decimal("0"))
    score: float = Field(default=0.0)

db_name = "gas_strategy.db"
engine = create_engine(f"sqlite:///{db_name}", echo=False)

def init_db():
    db_path = Path(f"{db_name}")
    if not db_path.exists():
        SQLModel.metadata.create_all(engine)

def analyze_window(session: Session, window_seconds: int):
    now = datetime.now()
    window_start = now - timedelta(seconds=window_seconds)

    statement = select(Measurement).where(Measurement.timestamp >= window_start, Measurement.timestamp <= now)
    results = session.exec(statement).all()
    N = len(results)
    if N == 0:
        return None
    sum_price = sum(float(m.price) for m in results)
    sum_score = sum(m.score for m in results)
    sum_duration = sum(m.duration.total_seconds() for m in results if m.duration)
    avg_price = sum_price / N
    score_mean = (sum_score / sum_duration) if sum_duration > 0 else 0.0
    return (avg_price / 1e9, score_mean / 1e9)

def update_price_tomlkit(path, section, key, value):
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc = parse(f.read())
        if section not in doc:
            doc[section] = {}
        doc[section][key] = int(value)
        with open(path, "w", encoding="utf-8") as f:
            f.write(dumps(doc))
        return True
    except Exception as e:
        LogPrint.error(f"[TOML ERROR]\n===\n{e}\n===")
        return False

def cycle_strategy_loop(
    loop: bool,
    base_price: int,
    time_params,
    multiplier_params,
    toml_path: str,
    mode: str,
    reverse_time: bool = False,
):
    import random
    min_time, max_time, time_vol = time_params
    min_mult, max_mult, _ = multiplier_params

    cycle_type = True  # True: long(low), False: short(high)
    while True:
        multiplier = random.uniform(min_mult, max_mult)
        long_time = max_time * (1 + random.uniform(-time_vol, time_vol))
        short_time = min_time * (1 + random.uniform(-time_vol, time_vol))
        period_time = long_time if cycle_type else short_time

        is_reverse = not cycle_type if reverse_time else cycle_type
        price = base_price if is_reverse else int(base_price * multiplier)

        success = update_price_tomlkit(toml_path, mode, "price", price)
        LogPrint.info(
            f"[{'OK' if success else 'NO'}] [{'LONG' if cycle_type else 'SHORT'}]\t[time={period_time:.3f}s]\t[price={price / 1e9 :.3f} gwei]"
        )

        time.sleep(period_time)

        with Session(engine) as session:
            measurement = Measurement(
                timestamp=datetime.now(),
                duration=timedelta(seconds=period_time),
                price=Decimal(price),
                score= period_time * price
            )
            session.add(measurement)
            session.commit()

            minute = analyze_window(session, 60)
            hour = analyze_window(session, 3600)
            day = analyze_window(session, 86400)
            if minute and hour and day:
                LogPrint.info(
                    f"- [ANALYSIS] [minute: ({minute[0]:.3f}, {minute[1]:.3f})]\t[hour: ({hour[0]:.3f}, {hour[1]:.3f})]\t[day: ({day[0]:.3f}, {day[1]:.3f})]"
                )
        cycle_type = not cycle_type
        if not loop:
            break


def main():
    init_db()
    parser = argparse.ArgumentParser(description="Price cycle strategy")
    parser.add_argument("--loop", "-l", action="store_true", help="Enable loop")
    parser.add_argument("--base-price", "-b", type=float, default=0.012, help="Base price (wei)")
    parser.add_argument("--time", "-t", type=float, nargs=3, metavar=('MIN', 'MAX', 'VOLATILITY'), required=True, help="Time: min max volatility (0~1)")
    parser.add_argument("--multiplier", "-m", type=float, nargs=3, metavar=('MIN', 'MAX', 'VOLATILITY'), required=True, help="Multiplier: min max volatility (0~1)")
    parser.add_argument("--toml-path", "-T", type=str, required=True, help="TOML file path")
    parser.add_argument("--mode", type=str, choices=["flashblocks", "market"], required=True, help="Write mode")
    parser.add_argument("--reverse-time", "-r", action="store_true", help="Reverse high/low time duration")
    args = parser.parse_args()

    LogPrint.info(f">>> [SET PRICE: {args.base_price} gwei | {int(args.base_price * 1e9)} wei] TOML={args.toml_path}, MODE={args.mode}")
    cycle_strategy_loop(
        loop=args.loop,
        base_price=int(args.base_price * 1e9),  # Convert Gwei to Wei
        time_params=args.time,
        multiplier_params=args.multiplier,
        toml_path=args.toml_path,
        mode=args.mode,
        reverse_time=args.reverse_time
    )

if __name__ == "__main__":
    main()
