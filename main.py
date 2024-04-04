import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import os
from tqdm import tqdm
import wrds

logging.basicConfig(filename='journal.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def retrieve_data(use_cached=True):
    file_name = "financial_data.xlsx"
    if os.path.exists(file_name) and use_cached:
        use_cache = input(
            f"Le fichier {file_name} existe déjà. Voulez-vous utiliser les données en cache ? (y/n): ").lower()
        if use_cache == 'y':
            logging.info("Chargement des données à partir du fichier Excel local.")
            return pd.read_excel(file_name)
    db = wrds.Connection()
    # Retrieve S&P 500 index membership from CRSP
    # You would replace 'your_username' and 'your_password' with your actual WRDS credentials
    for i in tqdm(range(2), desc="Récupération des données"):
        logging.info("Starting the main process...")
        pass
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
            SELECT a.gvkey, a.datadate, a.conm, b.sic, b.gsector, b.gsubind, a.ni, a.csho as shares_outsanding, a.at AS total_investments, a.lt AS total_liabilities, a.oiadp AS operating_income, a.dp AS depreciation_amortization, a.act AS current_assets, a.lct AS current_liabilities, a.dltt AS total_debt, a.sale AS total_revenues
            FROM comp.funda AS a
            JOIN comp.company AS b ON a.gvkey = b.gvkey                           
            WHERE a.gvkey IN ('{gvkeys_str}')
            AND a.datadate BETWEEN '2015-01-01' AND '2023-12-31'
            AND a.indfmt = 'INDL'
            AND a.datafmt='STD' 
            AND a.popsrc='D' 
            AND a.consol='C' 
            ORDER BY a.gvkey, a.datadate
            """)

        logging.info(f"Processing data : {retrieve_data_sql}")
        logging.info("All datas are recovered")
        # Exécutez la requête pour obtenir les données de cash flow
        dfs = [retrieve_data_sql]
        final_df = dfs[0]
        for df in dfs[1:]:
            final_df = pd.merge(final_df, df, on=['gvkey', 'datadate', 'conm', 'sic'], how='outer')
        # Exporter le DataFrame final en fichier Excel
    final_df.to_excel('financial_data.xlsx', index=False)
    logging.info("Excel file created")
    return pd.read_excel(file_name)


def ratio_calculation(final_df):
    # Calcul de Capitaux Propres comme différence entre Total des Actifs (`total_investments`) et Total des Passifs (
    # `total_liabilities`)
    final_df['equity'] = final_df['total_investments'] - final_df['total_liabilities']
    # ROA
    final_df['roa'] = final_df['ni'] / final_df['total_investments']
    # ROE
    final_df['roe'] = final_df['ni'] / final_df['equity']
    # Ratio de Liquidité Courante
    final_df['current_ratio'] = final_df['current_assets'] / final_df['current_liabilities']
    # Ratio d'Endettement
    final_df['debt_ratio'] = final_df['total_debt'] / final_df['total_investments']
    # Marge Opérationnelle
    final_df['operating_margin'] = final_df['operating_income'] / final_df['total_revenues']
    # Gestion des valeurs infinies ou manquantes après les calculs
    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['roa', 'roe', 'current_ratio', 'debt_ratio', 'operating_margin'], inplace=True)

    return final_df


def filter_companies_with_data_for_2023(df):
    # Convertir la colonne 'datadate' en datetime si ce n'est pas déjà fait
    df['datadate'] = pd.to_datetime(df['datadate'])

    # Identifier les gvkeys des entreprises ayant des données pour le 31 décembre 2023
    gvkeys_with_data_for_2023 = df[df['datadate'] == '2023-12-31']['gvkey'].unique()

    # Filtrer le DataFrame pour ne conserver que les données des entreprises identifiées
    filtered_df = df[df['gvkey'].isin(gvkeys_with_data_for_2023)]

    return filtered_df


def remove_outliers(df, group_field, field):
    # Group the data by the SIC code and calculate Z-scores within each group
    df['z_score'] = df.groupby(group_field)[field].transform(lambda x: zscore(x, ddof=1))

    # Define a threshold for identifying outliers
    z_threshold = 3

    # Remove outliers
    df_no_outliers = df[df['z_score'].abs() <= z_threshold]

    # Drop the z-score column as it's no longer needed
    df_no_outliers = df_no_outliers.drop('z_score', axis=1)

    return df_no_outliers


final_df = retrieve_data()

# Appliquer le filtrage
df_filtered = filter_companies_with_data_for_2023(final_df)

# Calcul des ratios financiers pour les entreprises filtrées
df_with_ratio = ratio_calculation(df_filtered)


ratios_to_clean = ['roa','roe', 'current_ratio', 'debt_ratio', 'operating_margin']
for ratio in ratios_to_clean:
    df_no_outliers = remove_outliers(df_with_ratio, 'sic', ratio)
df_no_outliers.to_excel('financial_data.xlsx', index=False)

logging.info("End of programme ")
