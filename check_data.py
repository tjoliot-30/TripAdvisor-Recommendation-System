import pandas as pd

try:
    df_places = pd.read_csv('Tripadvisor.csv', nrows=50)
    with open('check_data_out_utf8.txt', 'w', encoding='utf-8') as f:
        f.write("Places columns: " + str(df_places.columns.tolist()) + "\n")
except Exception as e:
    with open('check_data_out_utf8.txt', 'w', encoding='utf-8') as f:
        f.write(str(e))
