import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import wrds

def retrieve_data() :
    db = wrds.Connection()
    # Retrieve S&P 500 index membership from CRSP
    # You would replace 'your_username' and 'your_password' with your actual WRDS credentials
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
    # Créez une liste des gvkeys uniques
    gvkey_list = compustat_data['gvkey'].unique().tolist()
    gvkeys_str = "', '".join(gvkey_list)
    # Interrogez Compustat pour obtenir les cash flows opérationnels
    net_income_data = db.raw_sql("""
        SELECT tic, datadate, ni 
        FROM comp.funda 
        WHERE indfmt='INDL' 
        AND datafmt='STD' 
        AND popsrc='D' 
        AND consol='C' 
        AND datadate='2020-12-31'
        """)
    print(net_income_data)
    cash_flows_query = f"""
        SELECT gvkey, datadate, fqtr, oancfy
        FROM comp.fundq
        WHERE gvkey IN ('{gvkeys_str}')
        AND datadate BETWEEN '2015-01-01' AND '2023-12-31'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
        ORDER BY gvkey, datadate
        """
    shares_outstanding_data = db.raw_sql("""
        SELECT tic, datadate, csho
        FROM comp.funda
        WHERE indfmt='INDL' 
        AND datafmt='STD' 
        AND popsrc='D' 
        AND consol='C' 
        AND datadate BETWEEN '2020-01-01' AND '2020-12-31'
    """)
    print(shares_outstanding_data.head())
    total_investments_data = db.raw_sql("""
    SELECT tic, datadate, at AS total_investments
    FROM comp.funda
    WHERE indfmt='INDL' 
    AND datafmt='STD' 
    AND popsrc='D' 
    AND consol='C' 
    AND datadate BETWEEN '2020-01-01' AND '2020-12-31'
    """)
    print(total_investments_data.head())
    total_liabilities_data = db.raw_sql("""
    SELECT tic, datadate, lt AS total_liabilities
    FROM comp.funda
    WHERE indfmt='INDL' 
    AND datafmt='STD' 
    AND popsrc='D' 
    AND consol='C' 
    AND datadate BETWEEN '2020-01-01' AND '2020-12-31'
    """)
    print(total_liabilities_data.head())
    operating_income_data = db.raw_sql("""
    SELECT tic, datadate, oiadp AS operating_income
    FROM comp.funda
    WHERE indfmt='INDL' 
    AND datafmt='STD' 
    AND popsrc='D' 
    AND consol='C' 
    AND datadate BETWEEN '2020-01-01' AND '2020-12-31'
    """)
    print(operating_income_data.head())
    depreciation_amortization_data = db.raw_sql("""
    SELECT tic, datadate, dp AS depreciation_amortization
    FROM comp.funda
    WHERE indfmt='INDL' 
    AND datafmt='STD' 
    AND popsrc='D' 
    AND consol='C' 
    AND datadate BETWEEN '2020-01-01' AND '2020-12-31'
    """)
    print(depreciation_amortization_data())
    
    # Exécutez la requête pour obtenir les données de cash flow
    cash_flows_data = db.raw_sql(cash_flows_query)
    final_data = pd.merge(cash_flows_data, compustat_data[['gvkey', 'conm']], on='gvkey', how='left')
    final_data_cleaned = final_data.dropna(subset=['oancfy'])
    return tickers_str, final_data_cleaned, gvkey_list

retrieve_data() 