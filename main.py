import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import logging
import wrds

logging.basicConfig(filename='journal.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')
def retrieve_data() :
    db = wrds.Connection()
    # Retrieve S&P 500 index membership from CRSP
    # You would replace 'your_username' and 'your_password' with your actual WRDS credentials
    logging.info("Starting the main process...")
    mse = db.raw_sql("""
                        select comnam, ncusip, namedt, nameendt, 
                        permno, shrcd, exchcd, hsiccd, ticker
                        from crsp.msenames
                        """, date_cols=['namedt', 'nameendt'])
    # Créez une liste des tickers uniques
    ticker_list = mse['ticker'].dropna().unique().tolist()
    # Élimination des doublons en convertissant la liste en un ensemble, puis reconvertir en liste
    ticker_list = list(set(ticker_list))
    # Convertissez la liste des tickers en une chaîne pour la requête SQL
    tickers_str = "', '".join(ticker_list)
    # Exécutez votre requête pour obtenir les gvkeys des entreprises
    compustat_data = db.raw_sql("""
                        SELECT gvkey, conm
                        FROM comp.company
                        """)
    logging.info("Starting the gvkey retrieving...")
    # Créez une liste des gvkeys uniques
    gvkey_list = compustat_data['gvkey'].unique().tolist()
    gvkeys_str = "', '".join(gvkey_list)
    logging.info(f"Processing gvkey: {gvkeys_str}")
    logging.info("All gvkeys are recovered")
    # Interrogez Compustat pour obtenir les cash flows opérationnels
    logging.info(f"Starting the data retrieving...")
    retrieve_data_sql = db.raw_sql(f"""
        SELECT gvkey, datadate, conm, ni, csho, at as total_investments, lt AS total_liabilities, oiadp AS operating_income, dp AS depreciation_amortization
        FROM comp.funda 
        WHERE gvkey IN ('{gvkeys_str}')
        AND datadate BETWEEN '2015-01-01' AND '2023-12-31'
        AND indfmt = 'INDL'
        AND datafmt='STD' 
        AND popsrc='D' 
        AND consol='C' 
        ORDER BY gvkey, datadate
        """)
    print(retrieve_data_sql)
    logging.info(f"Processing data : {retrieve_data_sql}")
    logging.info("All datas are recovered")
    # Exécutez la requête pour obtenir les données de cash flow
    dfs = [retrieve_data_sql]
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on=['gvkey','datadate','conm'], how='outer')
    # Exporter le DataFrame final en fichier Excel
    final_df.to_excel('financial_data.xlsx', index=False)
    logging.info("Excel file created")

retrieve_data()
logging.info("End of programme ")
