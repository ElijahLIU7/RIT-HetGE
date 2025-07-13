import argparse
import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 配置Chrome浏览器
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")


def get_plddt_value(driver, name):
    """获取指定蛋白质名称的Average pLDDT值"""
    try:
        url = f"https://alphafold.com/search/text/{name}"
        driver.get(url)

        wait = WebDriverWait(driver, 20)
        element = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, '//div[contains(text(), "Average pLDDT")]')
            )
        )

        value_element = element.find_element(By.XPATH, "following-sibling::div")
        return value_element.text.strip().split()[0]

    except Exception as e:
        print(f"处理 {name} 时出错: {e}")
        return None


def process_csv(input_csv, success_file, failed_file, start_row=1, end_row=None):

    # 创建结果文件
    with open(success_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['protein_name', 'average_plddt'])

    with open(failed_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['protein_name', 'error_reason'])

    # 初始化浏览器驱动
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

    success_count = 0
    failed_count = 0
    current_row = 0  # 当前处理的行数

    try:
        with open(input_csv, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                current_row += 1

                logical_row = current_row

                # Determine whether it is within the specified number of lines
                if logical_row < start_row or logical_row > end_row:
                    continue

                if not row:
                    continue

                name = row[0].strip()
                if not name:
                    continue

                print(f"Processing row {logical_row}: {name}")

                plddt_value = get_plddt_value(driver, name)

                if plddt_value:
                    with open(success_file, 'a', newline='', encoding='utf-8') as outfile:
                        outfile.write(f"{name},{plddt_value}\n")
                    success_count += 1
                    print(f"✓ Success: {name} → {plddt_value}")
                else:
                    with open(failed_file, 'a', newline='', encoding='utf-8') as outfile:
                        outfile.write(f"{name}, No pLDDT value found.\n")
                    failed_count += 1
                    print(f"✗ Failure: {name}")

                time.sleep(1)  # request interval

    finally:
        driver.quit()
        print(f"\nProcessing complete (row range: {start_row} to {end_row or 'end'}).\n"
              f"Success count: {success_count} 条\n"
              f"Failure count: {failed_count} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='train')

    args = parser.parse_args()

    TrainOrTest = args.dataset
    INPUT_CSV = f"data/HGEProTstab/preprocess/{TrainOrTest}_filenames.csv"
    SUCCESS_FILE = f"data/HGEProTstab/preprocess/{TrainOrTest}_plddt_success.csv"
    FAILED_FILE = f"data/HGEProTstab/preprocess/{TrainOrTest}_plddt_failed.csv"

    # 配置行数参数
    START_ROW = 1  # 起始行
    END_ROW = 5000  # 结束行

    process_csv(INPUT_CSV, SUCCESS_FILE, FAILED_FILE, START_ROW, END_ROW)
