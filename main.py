import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
        logging.info(f"Starting the data retrieving...")
        retrieve_data_sql = db.raw_sql(f"""
            SELECT a.gvkey, a.iid, a.tic, a.datadate, a.conm, a.exchg, b.sic, b.gsector, b.gsubind, c.prccm AS stock_price, a.ni,a.txt, a.csho as shares_outsanding, a.dltt AS long_term_debt, a.at AS total_investments, a.lt AS total_liabilities, a.oiadp AS operating_income, a.dp AS depreciation_amortization, a.act AS current_assets, a.lct AS current_liabilities, a.dltt AS total_debt, a.sale AS total_revenues
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
        final_df = dfs[0]
        for df in dfs[1:]:
            final_df = pd.merge(final_df, df, on=['gvkey','iid' 'datadate', 'conm', 'sic'], how='outer')
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
    final_df['firm_value']= final_df['equity']+final_df['long_term_debt']
    #EBITDA
    final_df['EBITDA']= final_df['operating_income']+final_df['depreciation_amortization']
    #Market Cap
    final_df['Mcap']= final_df['shares_outsanding']*final_df['stock_price']

    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.dropna(subset=['roa', 'roe', 'current_ratio', 'debt_ratio', 'operating_margin','firm_value',"EBITDA","Mcap"], inplace=True)

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


final_df = retrieve_data()

# Appliquer le filtrage
df_filtered = filter_companies_with_data_for_2023(final_df)

# Calcul des ratios financiers pour les entreprises filtrées
df_with_ratio = ratio_calculation(df_filtered)

ratios_to_clean = ['roa', 'roe', 'current_ratio', 'debt_ratio', 'operating_margin']
for ratio in ratios_to_clean:
    df_no_outliers = remove_outliers(df_with_ratio, 'sic', ratio)
df_no_outliers.to_excel('financial_data.xlsx', index=False)


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


# Dropping specified columns to prepare explanatory variables X
columns_to_drop = [
    'roa', 'gvkey', 'datadate', 'conm', 'sic', 'gsector', 'gsubind', 'shares_outsanding'
]
X = df_no_outliers.drop(columns=columns_to_drop)

# Preparing the target variable y (column 'roa')
y = df_no_outliers['roa']

# Making sure all data in X is numeric and finite
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
X.replace([np.inf, -np.inf], 0, inplace=True)

# Cleaning the target variable y if necessary
y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

# Ensure X and y are of the same length
assert len(X) == len(y), "X and y must have the same number of rows"

# Calculate VIF for each explanatory variable
vif = pd.DataFrame()
vif["features"] = X.columns
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Display the VIF for each variable
print(vif)

# Exclude variables with a high VIF (e.g., VIF > 5)
high_vif_columns = vif[vif["VIF Factor"] > 5]["features"].tolist()
X_reduced = X.drop(columns=high_vif_columns)

# Apply stepwise selection and model construction with the refined set of variables
selected_features = stepwise_selection(X_reduced, y, verbose=True)

# Display selected variables
print(f"Selected features: {selected_features}")

# Construct the final model with selected features
X_final = sm.add_constant(X_reduced[selected_features])
model = sm.OLS(y, X_final).fit()

# Print the summary of the regression model
print(model.summary())

# Tracer les valeurs prédites par rapport aux valeurs réelles
y_pred = model.predict(X_final)

plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, alpha=0.3)  # alpha pour la transparence des points
plt.plot(y, y, color="red")  # Une ligne représentant le modèle parfait
plt.title('Valeurs prédites vs Valeurs réelles')
plt.xlabel('Valeurs réelles de ROA')
plt.ylabel('Valeurs prédites de ROA')
plt.show()
# Tracer les résidus du modèle
residuals = y - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color="red")
plt.title('Diagramme des résidus')
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.show()


# EDA(df_no_outliers)
logging.info("End of programme ")
