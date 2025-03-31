from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import time


# 設定 WebDriver
options = webdriver.ChromeOptions()
options.add_experimental_option("prefs", {
    "download.default_directory": os.path.abspath("downloads"),  # 設定下載路徑
    "download.prompt_for_download": False,  # 取消下載提示
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})

options.binary_location = "/usr/bin/google-chrome"
service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service, options=options)


service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 10)

# 1. 進入 NYSE 公司列表頁面
driver.get("https://www.responsibilityreports.com/Companies?exch=1")
time.sleep(3)  # 等待網頁載入

# 2. 爬取所有公司連結
companies = driver.find_elements(By.CSS_SELECTOR, "div a")
company_links = [c.get_attribute("href") for c in companies if c.get_attribute("href")]

for link in company_links:
    driver.get(link)
    time.sleep(3)  # 等待載入公司報告頁面
    
    # 3. 找到2022、2023、2024年的報告下載按鈕
    for year in ["2022", "2023", "2024"]:
        try:
            download_button = wait.until(EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{year} Global Impact Report')]/following-sibling::div/a[contains(text(), 'Download')]")))
            download_button.click()
            time.sleep(5)  # 等待下載完成
        except:
            print(f"公司 {link} 沒有 {year} 的報告")

# 結束 WebDriver
driver.quit()
