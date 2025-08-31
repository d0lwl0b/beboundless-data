# snapshot_tool.py
from botasaurus.browser import browser, Driver
from botasaurus import bt
from pathlib import Path

URLS = {
    "requestors_example": "https://explorer.beboundless.xyz/requestors",
    "requestor_example": "https://explorer.beboundless.xyz/requestors/0x2546c553d857d20658ece248f7c7d0861a240681?from=requestors",
    "order_example": "https://explorer.beboundless.xyz/orders/0x2546c553d857d20658ece248f7c7d0861a240681eb12d581?tab=more-details",
    "basescan_example": "https://basescan.org/tx/0x54e514a12b92f982af1bafbfe50c1d40417898258a3b2a1e2b70f764e4c2283e",
}
SNAPSHOT_DIR = Path(__file__).parent / "snapshot"
SNAPSHOT_DIR.mkdir(exist_ok=True)

@browser(
    parallel=False,
    cache=False,
    window_size=(1280, 720),
    wait_for_complete_page_load=True,
    create_error_logs=True,
    max_retry=3,
    retry_wait=5,
)
def save_dom_snapshot(driver: Driver, data):
    key = data["key"]
    url = data["url"]
    if key == "basescan_example":
        driver.google_get(url, bypass_cloudflare=True)
        driver.enable_human_mode()
        driver.long_random_sleep()
    else:
        driver.get(url)
        driver.enable_human_mode()
        driver.long_random_sleep()
    html_content = driver.page_html
    path = SNAPSHOT_DIR / f"{key}.html"
    bt.write_html(html_content, str(path))
    return {"key": key, "url": url, "saved_path": str(path)}

if __name__ == "__main__":
    data_list = [{"key": k, "url": u} for k, u in URLS.items()]
    results = save_dom_snapshot(data_list)
    print(results)
