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
        f"&offset=1"
        f"&sort=desc"
        f"&apikey={ETHERSCAN_API_KEY}"
    )
    response = request.get(url)
    if response.status_code == 404:
        raise NotFoundException(address)
    response.raise_for_status()
    data = response.json()
    if data.get("status") == "1" and data.get("result"):
        tx = data["result"][0]
        gas_price_wei = int(tx.get("gasPrice", "0"))
        return gas_price_wei / 1e9
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
import argparse

def main(loop=False, interval=60):
    while True:
        with Session(engine) as session:
            for address in MONITOR_ADDRESSES:
                try:
                    gas_price = fetch_latest_gas_price(address)
                except Exception as e:
                    LogPrint.error(f"Exception for {address}: {e}")
                    continue
                if gas_price is not None:
                    upsert_address_gas(session, address, gas_price)
                    LogPrint.info(f"{address} new gasPrice {gas_price:.2f} gwei")
                else:
                    LogPrint.error(f"Failed to fetch gas price for {address}")
        if not loop:
            break
        sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", help="Enable periodic monitoring loop")
    parser.add_argument("--interval", type=int, default=60, help="Loop interval in seconds")
    args = parser.parse_args()
    main(loop=args.loop, interval=args.interval)
