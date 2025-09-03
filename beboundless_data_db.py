# beboundless_data_db.py
from dotenv import load_dotenv
load_dotenv()

import re
import os
import sys
import time
import random
import threading
from datetime import datetime, timedelta, timezone
from string import Template
from functools import wraps
from pathlib import Path

class Settings:
    CYCLE_TIME = 1 * 60 * 60        # times in seconds
    ACTIVE_REQUESTOR_HOURS = 12
    RECENT_ORDER_HOURS = 2

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

def iso8601_to_utc_timestamp(iso_string: str) -> int:
    if iso_string.endswith('Z'):
        iso_string = iso_string.replace('Z', '+00:00')
    dt_object = datetime.fromisoformat(iso_string)
    return int(dt_object.timestamp())

from diskcache import Cache
from sqlmodel import Field, Relationship, SQLModel, Session, create_engine, select, col

cache = Cache()

db_name = "beboundless_data.db"
sqlite_url = f"sqlite:///{db_name}"
engine = create_engine(sqlite_url)

def with_session(engine):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Session(engine) as session:
                return func(*args, session=session, **kwargs)
        return wrapper
    return decorator

def init_db():
    db_path = Path(f"{db_name}")
    if not db_path.exists():
        SQLModel.metadata.create_all(engine)

class Requestors(SQLModel, table=True):  
    address: str = Field(primary_key=True)
    orders_requested: int  
    cycles_requested: int  
    last_activity: int  
    avg_wei_per_cycle: float
    requests_acceptance_rate: float  
    orders: list["Orders"] = Relationship(back_populates="requestor")  
  
class Orders(SQLModel, table=True):  
    order_id: str = Field(primary_key=True)  
    requestor_address: str = Field(foreign_key="requestors.address")
    status: str
    created: int
    lock_txn: str | None = Field(default=None)  
    lock_value: float | None = Field(default=None)  
    lock_transaction_fee: float | None = Field(default=None)  
    lock_gas_price: float | None = Field(default=None)  
    requestor: Requestors | None = Relationship(back_populates="orders")

def upsert_requestors(session: Session, data: list[tuple]):
    for item in data:
        address, orders_requested, cycles_requested, last_activity, avg_wei_per_cycle, requests_acceptance_rate = item
        obj = session.get(Requestors, address)
        fields = {
            "address": address,
            "orders_requested": int(orders_requested),
            "cycles_requested": int(cycles_requested),
            "last_activity": int(last_activity),
            "avg_wei_per_cycle": float(avg_wei_per_cycle),
            "requests_acceptance_rate": float(requests_acceptance_rate.rstrip('%')) / 100 if isinstance(requests_acceptance_rate, str) and requests_acceptance_rate.endswith('%') else float(requests_acceptance_rate)
        }
        if obj:
            obj.sqlmodel_update(fields)
            session.add(obj)
        else:
            obj = Requestors(**fields)
            session.add(obj)
    session.commit()

def upsert_orders(session: Session, data: list[dict]):
    for result in data:
        for address, orders in result.items():
            requestor_obj = session.get(Requestors, address)
            if not requestor_obj:
                LogPrint.warning(f"Requestor {address} not found for orders, skipping.")
                continue
            for order_id, status, created in orders:
                obj = session.get(Orders, order_id)
                fields = {
                    "order_id": order_id,
                    "requestor_address": address,
                    "status": status,
                    "created": created,
                }
                if obj:
                    obj.sqlmodel_update(fields)
                    obj.requestor = requestor_obj
                    session.add(obj)
                else:
                    obj = Orders(**fields)
                    obj.requestor = requestor_obj
                    session.add(obj)
                if obj not in requestor_obj.orders:
                    requestor_obj.orders.append(obj)
    session.commit()

def upsert_lock_txn(session: Session, data: dict):
    """
    data: {order_id: (lock_txn, lock_value, lock_transaction_fee, lock_gas_price)}
    """
    for order_id, (lock_txn, lock_value, lock_transaction_fee, lock_gas_price) in data.items():
        obj = session.get(Orders, order_id)
        if not obj:
            LogPrint.warning(f"Order {order_id} not found, skipping.")
            continue
        obj.lock_txn = lock_txn
        obj.lock_value = lock_value
        obj.lock_transaction_fee = lock_transaction_fee
        obj.lock_gas_price = lock_gas_price
        session.add(obj)
    session.commit()

def get_active_requestors(session, hours=None):
    hours = hours if hours is not None else Settings.ACTIVE_REQUESTOR_HOURS
    time_ago = int((datetime.now() - timedelta(hours=hours)).timestamp())
    statement = select(Requestors).where(Requestors.last_activity >= time_ago)
    return session.exec(statement).all()

def get_recent_non_submitted_order_ids(session: Session, hours=None) -> list[str]:
    hours = hours if hours is not None else Settings.RECENT_ORDER_HOURS
    time_ago = int((datetime.now() - timedelta(hours=hours)).timestamp())
    statement = select(Orders.order_id).where(
        Orders.status != "Submitted",
        Orders.created >= time_ago,
        Orders.lock_gas_price == None
    )
    return session.exec(statement).all()

from botasaurus.browser import browser, Driver, AsyncQueueResult, Wait
from botasaurus.request import request, Request
from botasaurus.soupify import soupify

def delete_old_orders(session: Session, days: int = 2):
    """Delete Orders older than `days` days."""
    cutoff = int((datetime.now() - timedelta(days=days)).timestamp())
    statement = select(Orders).where(Orders.created < cutoff)
    old_orders = session.exec(statement).all()
    for order in old_orders:
        session.delete(order)
    session.commit()

_REQUESTORS_LINK = "https://explorer.beboundless.xyz/requestors"
_REQUESTOR_LINK = Template("https://explorer.beboundless.xyz/requestors/${address}?from=requestors")
_ORDER_LINK = Template("https://explorer.beboundless.xyz/orders/${order_id}?tab=more-details")

@cache.memoize(expire=180)
@browser(block_images=True, window_size=(1280, 720))
def scrape_requestors(driver: Driver, data):
    driver.get(_REQUESTORS_LINK)
    driver.wait_for_element("table", wait=Wait.LONG)
    soup = soupify(driver)
    table = soup.select_one("table")
    rows = table.select("tbody tr") if table else []
    LogPrint.info(f"Found {len(rows)} requestors")

    if not rows:
        LogPrint.error("No requestors found")
        return None

    result = []
    for row in rows:
        cells = row.select("td")
        if len(cells) != 6:
            LogPrint.warning(f"Unexpected number of cells in row: {len(cells)}")
            continue

        requestor_address = cells[0].select_one("a span")["title"]
        orders_requested = cells[1].select_one("p")["title"]
        cycles_requested = cells[2].select_one("p")["title"]
        last_activity = iso8601_to_utc_timestamp(cells[3].select_one("time")["datetime"])
        avg_wei_per_cycle = cells[4].select_one("p")["title"]
        requests_acceptance_rate = cells[5].text
        result.append((requestor_address, orders_requested, cycles_requested, last_activity, avg_wei_per_cycle, requests_acceptance_rate))
    return result

@cache.memoize(expire=180)
@browser(reuse_driver=True, block_images=True, window_size=(1280, 720))
def scrape_orders(driver: Driver, data):
    driver.get(_REQUESTOR_LINK.substitute(address=data))
    driver.wait_for_element("table", wait=Wait.LONG)
    soup = soupify(driver)

    table = soup.select_one("table")
    rows = table.select("tbody tr") if table else []
    LogPrint.info(f"Found {len(rows)} orders")

    if not rows:
        LogPrint.error("No orders found")
        return None

    result = {data: []}
    for row in rows:
        cells = row.select("td")
        if len(cells) != 9:
            LogPrint.warning(f"Unexpected number of cells in row: {len(cells)}")
            continue
        order_id = cells[0].select_one("a span")["title"]
        status = cells[1].select_one("span").text
        created = iso8601_to_utc_timestamp(cells[4].select_one("time")["datetime"])
        result[data].append((order_id, status, created))

    return result

LOCK_TXN_REGEX = re.compile(r'(?:\\"lock_txn\\":\\"|lock_txn":")(?P<txn>0x[a-fA-F0-9]{64})')

@cache.memoize(expire=360)
@request(max_retry=10, retry_wait=1, parallel=5)
def scrape_lock_txn(request: Request, order_id: str):
    time.sleep(random.uniform(0.3, 0.8))
    response = request.get(_ORDER_LINK.substitute(order_id=order_id))
    response.raise_for_status()
    match = LOCK_TXN_REGEX.search(response.text)
    if match:
        lock_txn = match.group("txn")
        return (order_id, lock_txn)
    else:
        LogPrint.warning(f"lock_txn not found for order {order_id}")
        return (order_id, None)

_last_request_time = 0
_request_lock = threading.Lock()

@cache.memoize(expire=360)
@request(max_retry=10, retry_wait=2, parallel=1)
def scrape_lock_info(request: Request, txhash: str):
    global _last_request_time
    with _request_lock:
        current_time = time.time()
        time_since_last = current_time - _last_request_time
        if time_since_last < 0.3:
            time.sleep(0.3 - time_since_last)
        _last_request_time = time.time()

    hex_to_eth = lambda hex_str: int(hex_str, 16) / 1e18
    hex_to_gwei = lambda hex_str: int(hex_str, 16) / 1e9
    API_KEY = os.environ["BASESCAN_API_KEY"]
    url_tx = f"https://api.etherscan.io/v2/api?chainid=8453&module=proxy&action=eth_getTransactionByHash&txhash={txhash}&apikey={API_KEY}"
    resp_tx = request.get(url_tx)
    data_tx = resp_tx.json().get("result", {})
    value = hex_to_eth(data_tx.get("value", "0x0"))
    gas_price = hex_to_gwei(data_tx.get("gasPrice", "0x0"))
    return (txhash, value, gas_price)

@with_session(engine)
def main(session: Session, active_hours=None, order_hours=None):
    delete_old_orders(session, days=2)

    active_hours = active_hours if active_hours is not None else Settings.ACTIVE_REQUESTOR_HOURS
    order_hours = order_hours if order_hours is not None else Settings.RECENT_ORDER_HOURS

    data = scrape_requestors()
    upsert_requestors(session, data) if data else LogPrint.error("Failed to scrape requestors")

    active_requestors = get_active_requestors(session, hours=active_hours)
    LogPrint.info(f"Found {len(active_requestors)} active requestors")
    addresses = [r.address for r in active_requestors]

    results = scrape_orders(addresses)
    upsert_orders(session, results) if results else LogPrint.error("Failed to scrape orders")

    active_orders = get_recent_non_submitted_order_ids(session, hours=order_hours)
    LogPrint.info(f"Found {len(active_orders)} active orders")

    results = scrape_lock_txn(active_orders)
    results = [item for item in results if item[1] is not None]
    # results: list of (order_id, lock_txn)
    txns = scrape_lock_info([txn for _, txn in results])
    # txns: list of (txhash, value, gas_price)

    # Combine results and txns into {order_id: (lock_txn, lock_value, lock_transaction_fee, lock_gas_price)}
    orderid_to_txn = dict(results)
    txninfo = {txhash: (value, gas_price) for txhash, value, gas_price in txns}
    final_data = {}
    for order_id, lock_txn in orderid_to_txn.items():
        value, gas_price = txninfo.get(lock_txn, (None, None))
        final_data[order_id] = (lock_txn, value, None, gas_price)

    final_data
    upsert_lock_txn(session, final_data) if final_data else LogPrint.error("Failed to scrape lock transactions")

def periodic_job(active_hours=None, order_hours=None, cycle_time=None):
    cycle_time = cycle_time if cycle_time is not None else Settings.CYCLE_TIME
    main(active_hours=active_hours, order_hours=order_hours)
    LogPrint.info(f"Next update in {cycle_time//60} min")
    t = threading.Timer(cycle_time, lambda: periodic_job(active_hours=active_hours, order_hours=order_hours, cycle_time=cycle_time))
    t.daemon = True
    t.start()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--active_hours", type=int, help="Active requestor time window (hours)")
    parser.add_argument("--order_hours", type=int, help="Recent order time window (hours)")
    parser.add_argument("--cycle_time", type=int, help="Periodic job cycle time (seconds)")
    parser.add_argument("--go", action="store_true", help="Run in periodic mode")
    args = parser.parse_args()

    if args.active_hours is not None:
        Settings.ACTIVE_REQUESTOR_HOURS = args.active_hours
    if args.order_hours is not None:
        Settings.RECENT_ORDER_HOURS = args.order_hours
    if args.cycle_time is not None:
        Settings.CYCLE_TIME = args.cycle_time

    if args.go:
        LogPrint.info(f"Periodic mode started, update every {Settings.CYCLE_TIME//60} min")
        periodic_job(active_hours=args.active_hours, order_hours=args.order_hours, cycle_time=args.cycle_time)
        try:
            while True:
                time.sleep(Settings.CYCLE_TIME)
        except KeyboardInterrupt:
            LogPrint.info("Periodic mode stopped by user")
    else:
        main(active_hours=args.active_hours, order_hours=args.order_hours)
