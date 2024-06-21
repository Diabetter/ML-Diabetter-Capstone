from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
import locale

import pandas as pd

# Set locale
locale.setlocale(locale.LC_ALL, '')

# Initialize Chrome driver
chrome = webdriver.Chrome()
chrome.get("https://www.panganku.org/en-EN/semua_nutrisi")
chrome.implicitly_wait(10)

# Declare list of dictionary for string data result
foods = []


def get_food_data(driver, table_row):
    data = {}
    field_names = ["Food Code", "Food Name", "Group", "Type"]

    for i in range(4):
        element = driver.find_element(By.XPATH, f"//*[@id=\"data\"]/tbody/tr[{table_row}]/td[{i + 2}]")
        data[field_names[i]] = element.text

    return data


def get_nutrition_compositions_data(driver):
    data = {}

    table_of_nutritions = driver.find_elements(By.XPATH, '//*[@id="print-area"]/div[2]/section/div/div')

    for i in range(len(table_of_nutritions)):
        nutritions = table_of_nutritions[i].find_elements(By.XPATH, './div/div/table/tbody/tr')

        for j in range(len(nutritions)):
            result = list(map(lambda x: x.strip(), nutritions[j].text.split(":")))
            if len(result) == 2:
                value = result[1].split(" ")

                data[f"{result[0]} ({value[1]})"] = locale.atof(value[0])

    return data


# Show all data, so we can get all the data
data_length_select = Select(chrome.find_element(By.XPATH, '//*[@id="data_length"]/label/select'))
data_length_select.select_by_value("-1")

# Execute the scraping
table_of_foods = chrome.find_elements(By.XPATH, '//*[@id="data"]/tbody/tr')
for i in range(len(table_of_foods)):
    food = get_food_data(chrome, i + 1)

    data = chrome.find_element(By.XPATH, f"//*[@id='data']/tbody/tr[{i + 1}]")
    data.click()

    nutrition_compositions = get_nutrition_compositions_data(chrome)
    food.update(nutrition_compositions)
    # food_df.concat(food, ignore_index=True)
    foods.append(food)

    print(f"{i + 1}: {food['Food Code']} | {food['Food Name']}")

    chrome.back()

# Quit the Chrome
chrome.quit()

# Transform the dictionary to dataframe and save it
food_df = pd.DataFrame.from_records(foods)
food_df.to_csv("food_data.csv", index=False)

# %%
