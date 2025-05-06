import csv
import os

from playwright.sync_api import sync_playwright

URL_PREFIX = "https://www.responsibilityreports.com"
OUTPUT_FILE = "output.csv"


def get_company_name_and_ticker_name(exchanger: str):
    html_path = os.path.join(os.getcwd(), f"{exchanger}.html")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{html_path}")

        items = page.locator("li")
        count = items.count()

        write_header = (
            not os.path.exists(OUTPUT_FILE) or os.stat(OUTPUT_FILE).st_size == 0
        )
        with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["company_name", "exchange", "ticker"]
            )
            if write_header:
                writer.writeheader()
            # 如果是新檔案就寫 header（必要時）
            if os.stat(OUTPUT_FILE).st_size == 0:
                writer.writeheader()

            for i in range(count):
                item = items.nth(i)
                if item.locator(".companyName").count() == 0:
                    continue

                try:
                    company_name = item.locator(".companyName a").inner_text()
                    if company_name in processed_data:
                        print(f"已處理過 {company_name}，跳過")
                        continue

                    href = item.locator(".companyName a").get_attribute("href")
                    if href is None:
                        print(f"{company_name} 無效 href，跳過")
                        continue

                    print(
                        f"[{i+1} / {count}] 提取 {company_name} 公司股票代號...",
                        end="",
                        flush=True,
                    )

                    temp_page = browser.new_page()
                    temp_page.goto(f"{URL_PREFIX}{href}")

                    locator = temp_page.locator(".ticker_name")
                    locator.wait_for(timeout=3000)
                    ticker_name = locator.inner_text()

                    print("成功提取")
                    writer.writerow(
                        {
                            "company_name": company_name,
                            "exchange": exchanger,
                            "ticker": ticker_name,
                        }
                    )
                    temp_page.close()
                except:
                    print("無法提取，填入預設值（N/A）")
                    writer.writerow(
                        {
                            "company_name": company_name,
                            "exchange": exchanger,
                            "ticker": "N/A",
                        }
                    )
                    temp_page.close()


if __name__ == "__main__":
    processed_data = set()

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_data.add(row["company_name"])

    get_company_name_and_ticker_name("NYSE")
    get_company_name_and_ticker_name("NASDAQ")
