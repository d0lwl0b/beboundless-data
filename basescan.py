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

# Database
from sqlmodel import SQLModel, Field, Session, create_engine, select

DB_NAME = "address_gas_monitor.db"

class AddressGas(SQLModel, table=True):
    address: str = Field(primary_key=True)
    gas_price: float

engine = create_engine(f"sqlite:///{DB_NAME}")

if not Path(DB_NAME).exists():
    SQLModel.metadata.create_all(engine)

# Request
from scipy.stats import trim_mean
from botasaurus.request import request, Request, NotFoundException

@request(cache=False, max_retry=10, retry_wait=1, raise_exception=True, create_error_logs=True)
def fetch_latest_gas_price(request: Request, address: str) -> float | None:
    sleep(random.uniform(0.1, 0.3))  # To avoid rate limiting
    url = (
        "https://api.etherscan.io/v2/api"
        f"?chainid=8453"
        f"&module=account"
        f"&action=txlist"
        f"&address={address}"
        f"&page=1"
        f"&offset=3000"
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
        gas_prices = [int(tx.get("gasPrice", "0")) for tx in txs if tx.get("gasPrice")]
        if not gas_prices:
            return None
        stable_mean = trim_mean(gas_prices, 0.1)
        return int(stable_mean)
    return None

def upsert_address_gas(session: Session, address: str, gas_price: float):
    obj = session.get(AddressGas, address)
    if obj:
        obj.gas_price = gas_price
        session.add(obj)
    else:
        obj = AddressGas(address=address, gas_price=gas_price)
        session.add(obj)
    session.commit()


# Main
import math
from tomlkit import parse, dumps
import argparse

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

def main(loop=False, interval=60, toml_path=None, markup=1.07):
    while True:
        with Session(engine) as session:
            try:
                gas_prices = fetch_latest_gas_price(MONITOR_ADDRESSES)
            except Exception as e:
                LogPrint.error(f"Exception for {MONITOR_ADDRESSES}: {e}")
                gas_prices = [None] * len(MONITOR_ADDRESSES)
            valid_prices = []
            for address, gas_price in zip(MONITOR_ADDRESSES, gas_prices):
                if gas_price is not None:
                    upsert_address_gas(session, address, gas_price)
                    gas_price_gwei = gas_price / 1e9
                    LogPrint.info(f"{address} new gasPrice ({gas_price_gwei:.2f} gwei | {gas_price:.2f} wei)")
                    valid_prices.append(gas_price)
                else:
                    LogPrint.error(f"Failed to fetch gas price for {address}")
            if valid_prices and toml_path:
                # RMS (quadratic mean)
                rms = math.sqrt(sum(x**2 for x in valid_prices) / len(valid_prices))
                my_price = int(rms * markup)
                update_toml_price(toml_path, my_price)
                LogPrint.info(f"Final RMS price: {rms:.2f}, quoted price: {my_price} (markup={markup})")
        if not loop:
            break
        sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--loop", action="store_true", help="Enable periodic monitoring loop")
    parser.add_argument("-i", "--interval", type=int, default=60, help="Loop interval in seconds")
    parser.add_argument("-t", "--toml-path", type=str, required=True, help="Path to the toml file to update")
    parser.add_argument("-m", "--markup", type=float, default=1.07, help="Markup ratio for final price (e.g. 1.07 means +7%)")
    args = parser.parse_args()
    main(loop=args.loop, interval=args.interval, toml_path=args.toml_path, markup=args.markup)
