import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import logging
import os
from tqdm import tqdm
import wrds
import mplcursors


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
        logging.info(f"Starting the data retrieving...")
        retrieve_data_sql = db.raw_sql(f"""
            SELECT a.gvkey, a.iid, a.tic, a.datadate, a.conm, a.exchg, b.sic, b.gsector, b.gsubind, c.prccm AS stock_price, a.ni,a.txt, a.csho as shares_outstanding, a.dltt AS long_term_debt, a.at AS total_investments, a.lt AS total_liabilities, a.oiadp AS operating_income, a.dp AS depreciation_amortization, a.act AS current_assets, a.lct AS current_liabilities, a.dltt AS total_debt, a.sale AS total_revenues
            FROM comp.funda AS a
            JOIN comp.company AS b ON a.gvkey = b.gvkey 
            INNER JOIN comp.secm c
            on a.gvkey = c.gvkey
            and a.iid = c.iid
            and a.datadate = c.datadate
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
        # Exécutez la requête pour obtenir les données
        dfs = [retrieve_data_sql]
        df_copy = dfs[0]
        for df in dfs[1:]:
            df_copy = pd.merge(df_copy, df, on=['gvkey','iid' 'datadate', 'conm', 'sic'], how='outer')
        # Exporter le DataFrame final en fichier Excel
    df_copy.to_excel('financial_data.xlsx', index=False)
    logging.info("Excel file created")
    return pd.read_excel(file_name)


def ratio_calculation(df):
    # It's good practice to work on a copy to avoid changing the original dataframe
    df_copy = df.copy()

    # Calculating financial ratios and other columns
    df_copy.loc[:, 'equity'] = df_copy['total_investments'] - df_copy['total_liabilities']
    df_copy.loc[:, 'roa'] = df_copy['ni'] / df_copy['total_investments']
    df_copy.loc[:, 'roe'] = df_copy['ni'] / df_copy['equity']
    df_copy.loc[:, 'current_ratio'] = df_copy['current_assets'] / df_copy['current_liabilities']
    df_copy.loc[:, 'debt_ratio'] = df_copy['total_debt'] / df_copy['total_investments']
    df_copy.loc[:, 'operating_margin'] = df_copy['operating_income'] / df_copy['total_revenues']
    df_copy.loc[:, 'firm_value'] = df_copy['equity'] + df_copy['long_term_debt']
    df_copy.loc[:, 'EBITDA'] = df_copy['operating_income'] + df_copy['depreciation_amortization']
    df_copy.loc[:, 'Mcap'] = df_copy['shares_outstanding'] * df_copy['stock_price']
    df_copy.loc[:, 'E/D+E']= df_copy['equity']/(df_copy['long_term_debt']+df_copy['equity'])
    df_copy['EV/EBITDA'] = df_copy['firm_value'] / df_copy['EBITDA']
    df_copy['EV/EBITDA(1+tr)'] = df_copy['firm_value'] / (df_copy['EBITDA'] * (1 + df_copy['txt']))
    std_devs = df_copy.groupby('gvkey')['stock_price'].std()
    df_copy['SD_StockPrice'] = df_copy['gvkey'].map(std_devs)
    df_copy['EV/EBIT'] = df_copy['firm_value'] / df_copy['operating_income']
    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_copy.dropna(subset=['roa', 'roe', 'current_ratio', 'debt_ratio', 'operating_margin','firm_value',"EBITDA","Mcap","E/D+E","EV/EBITDA","EV/EBITDA(1+tr)","SD_StockPrice","EV/EBIT"], inplace=True)

    return df_copy


def filter_companies_with_data_for_2023(df):
    # Convertir la colonne 'datadate' en datetime si ce n'est pas déjà fait
    df['datadate'] = pd.to_datetime(df['datadate'])

    # Identifier les gvkeys des entreprises ayant des données pour l'année 2023
    gvkeys_with_data_for_2023 = df[df['datadate'].dt.year == 2023]['gvkey'].unique()

    # Trouver les gvkeys des entreprises qui n'ont PAS de données pour 2023
    gvkeys_without_data_for_2023 = df[~df['gvkey'].isin(gvkeys_with_data_for_2023)]['gvkey'].unique()

    # Filtrer pour exclure les entreprises identifiées sans données en 2023
    df = df[~df['gvkey'].isin(gvkeys_without_data_for_2023)]

    # Additionally, filter out rows where 'equity' is NA or zero
    df = df.dropna(subset=['equity'])
    df = df[df['equity'] != 0]

    # Filter out rows where ROE or ROA are above 200%
    df = df[(df['roe'] <= 2) & (df['roa'] <= 2)]

    return df

def EDA(df):
    # Convert the 'datadate' column to datetime
    df['datadate'] = pd.to_datetime(df['datadate'])

    # Summary statistics for numerical columns
    print(df.describe())
    # Calculate the median ROA for each year across all companies
    median_roa_by_year = df.groupby(df['datadate'].dt.year)['roa'].median().reset_index()

    # Plotting the median ROA trend
    plt.figure(figsize=(14, 7))
    sns.lineplot(x='datadate', y='roa', data=median_roa_by_year)
    plt.title('Median Trend of ROA for All Companies (2015-2023)')
    plt.xlabel('Year')
    plt.ylabel('Median Return on Investment')
    plt.show()

    # Histogramme avec ligne de densité pour 'roa'
    plt.figure(figsize=(14, 7))
    sns.histplot(df['roa'], bins=50, kde=True)
    plt.title('ROA Distribution with Density Line for All Companies (2015-2023)')
    plt.xlabel('Return on Assets')
    plt.show()

    # Empiled Boxplot avec histogramme pour 'roa'
    plt.figure(figsize=(14, 7))
    sns.histplot(df['roa'], element="step", fill=False, cumulative=False, bins=100, kde=True)
    sns.boxplot(x=df['roa'], whis=[0, 100], width=0.1, color=".7")
    plt.title('ROA Stacked Boxplot with Histogram for All Companies (2015-2023)')
    plt.xlabel('Return on Assets')
    plt.show()

    # CDF pour 'roa'
    plt.figure(figsize=(14, 7))
    sns.ecdfplot(df['roa'])
    plt.title('Cumulative Distribution Function of ROA for All Companies (2015-2023)')
    plt.xlabel('Return on Assets')
    plt.ylabel('CDF')
    plt.show()

    # Bee Swarm Plot pour 'roa'
    plt.figure(figsize=(14, 7))
    sns.swarmplot(y=df['roa'], size=2)
    plt.title('Bee Swarm Plot of ROA for All Companies (2015-2023)')
    plt.xlabel('Return on Assets')
    plt.show()

    # Correlation heatmap for numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_exclude = ['gvkey', 'sic', 'gsector', 'gsubind']  # Liste des colonnes à exclure
    cols_to_include = [col for col in numerical_cols if col not in cols_to_exclude]

    correlation_matrix = df[cols_to_include].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap for Financial Ratios (2015-2023)')
    plt.show()

    # Select a subset of ratios for the pair plot to keep it readable
    sample_df = df.sample(n=10000, random_state=1)  # Adjust n as needed
    selected_ratios = ['roa', 'roe', 'current_ratio', 'debt_ratio']
    sns.pairplot(sample_df[selected_ratios])
    plt.suptitle('Pair Plot of Selected Financial Ratios', y=1.02)  # Adjust y for title to display correctly
    plt.show()

    # Outlier detection
    Q1 = df['roa'].quantile(0.25)
    Q3 = df['roa'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = 1.5 * IQR
    df['roa_outlier'] = ((df['roa'] < (Q1 - outlier_threshold)) | (df['roa'] > (Q3 + outlier_threshold)))

    # Time series decomposition (assuming monthly data with annual seasonality)
    # Please adjust the 'period' parameter based on your data's frequency
    time_series = df.set_index('datadate')['roa']
    time_series = time_series.resample('M').median().dropna()  # Resampling to monthly frequency
    decomp = seasonal_decompose(time_series, model='additive', period=12)

    plt.figure(figsize=(14, 7))
    decomp.plot()
    plt.show()


def stepwise_selection(X, y, initial_list=[], threshold_in=0.04, threshold_out=0.1, verbose=True):
    """ Perform a forward-backward feature selection
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features
    """

    included = list(initial_list)
    while True:
        changed = False
        # forward step
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included


def plot_predicted_vs_real(df_copy, y_real, y_pred):
    """
        Trace un graphique interactif comparant les valeurs prédites aux valeurs réelles,
        avec une ligne représentant les prédictions parfaites et des annotations pour chaque point.

        Parameters:
        - df_copy: DataFrame contenant les données originales, utilisé pour extraire des informations supplémentaires pour les annotations.
        - y_real: Série pandas contenant les valeurs réelles.
        - y_pred: Série pandas contenant les valeurs prédites par le modèle.
        """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(y_real, y_pred, alpha=0.3, color='blue')  # alpha for point transparency
    plt.title('Valeurs Prédites vs Valeurs Réelles')
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Valeurs Prédites')

    # Trace la ligne y = x pour référence
    max_val = max(y_real.max(), y_pred.max())
    min_val = min(y_real.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Assurez-vous que 'conm' est la colonne avec les noms des entreprises
        # Ajustez 'conm' au nom de votre colonne si nécessaire
        sel.annotation.set(text=df_copy['conm'].iloc[sel.target.index],
                           position=(20, -20))  # Ajustez la position selon le besoin
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.6)

    plt.grid(True)
    plt.show()


def evaluate_model_performance(model, X, y):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    import seaborn as sns

    # Assurez-vous que X inclut la constante si votre modèle en dépend
    if not 'const' in X.columns:
        X = sm.add_constant(X)

    # Recalculer y_pred pour s'assurer de l'alignement avec y
    y_pred = model.predict(X)

    # Calcul du R² et du R² ajusté
    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {adjusted_r_squared}")

    # Calcul des résidus
    residuals = y - y_pred

    # Vérifiez que y_pred et y ont la même longueur (sécurité supplémentaire)
    assert len(y_pred) == len(y), "y_pred and y must have the same length"

    # Diagramme des résidus
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color="red")
    plt.title('Diagramme des résidus')
    plt.xlabel('Valeurs prédites')
    plt.ylabel('Résidus')
    plt.show()

    # Q-Q plot des résidus
    sm.qqplot(residuals, line='s')
    plt.title('Q-Q plot des résidus')
    plt.show()

    # Analyse de la distribution des résidus avec un histogramme
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution des résidus')
    plt.xlabel('Résidus')
    plt.show()

    # Test de Breusch-Pagan pour l'hétéroscédasticité
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(residuals, X)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print("Test de Breusch-Pagan pour l'hétéroscédasticité :", dict(zip(labels, bp_test)))

