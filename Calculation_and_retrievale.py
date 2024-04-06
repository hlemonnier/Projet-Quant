import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, jarque_bera
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import logging
import os
from tqdm import tqdm
import wrds
import mplcursors
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


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
        # Create a list of unique tickers
        ticker_list = mse['ticker'].dropna().unique().tolist()
        # Eliminate duplicates by converting the list into a set, then back into a list
        ticker_list = list(set(ticker_list))
        # Convert the list of tickers into a string for the SQL query
        tickers_str = "', '".join(ticker_list)
        # Execute your request to obtain the companies' gvkeys
        compustat_data = db.raw_sql("""
                            SELECT gvkey, conm
                            FROM comp.company
                            """)
        logging.info("Starting the gvkey retrieving...")
        # Create a list of unique gvkeys
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
        # Run the query to obtain the data
        dfs = [retrieve_data_sql]
        df_copy = dfs[0]
        for df in dfs[1:]:
            df_copy = pd.merge(df_copy, df, on=['gvkey','iid' 'datadate', 'conm', 'sic'], how='outer')
        # Export the final DataFrame as an Excel file
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
    # Convert the 'datadate' column to datetime if you haven't already done so
    df['datadate'] = pd.to_datetime(df['datadate'])

    # Identify the gvkeys of companies with data for the year 2023
    gvkeys_with_data_for_2023 = df[df['datadate'].dt.year == 2023]['gvkey'].unique()

    # Find the gvkeys of companies that have NO data for 2023
    gvkeys_without_data_for_2023 = df[~df['gvkey'].isin(gvkeys_with_data_for_2023)]['gvkey'].unique()

    # Filter to exclude companies identified as having no data in 2023
    df = df[~df['gvkey'].isin(gvkeys_without_data_for_2023)]

    # Additionally, filter out rows where 'equity' is NA or zero
    df = df.dropna(subset=['equity'])
    df = df[df['equity'] != 0]

    # Filter out rows where ROE or ROA are above 200%
    df = df[(df['roe'].abs() <= 25) & (df['roa'].abs() <= 25)]
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

    # Histogram with density line for 'roa
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

    # CDF for 'roa
    plt.figure(figsize=(14, 7))
    sns.ecdfplot(df['roa'])
    plt.title('Cumulative Distribution Function of ROA for All Companies (2015-2023)')
    plt.xlabel('Return on Assets')
    plt.ylabel('CDF')
    plt.show()

    # Bee Swarm Plot for 'roa
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
        Plots an interactive graph comparing predicted values with actual values,
        with a line representing perfect predictions and annotations for each point.
        Parameters:
        - df_copy: DataFrame containing the original data, used to extract additional information for annotations.
        - y_real: Pandas series containing real values.
        - y_pred: Pandas series containing the values predicted by the model.
        """
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(y_real, y_pred, alpha=0.3, color='blue')  # alpha for point transparency
    plt.title('Predicted values vs. actual values')
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')

    # Draw the line y = x as a reference
    max_val = max(y_real.max(), y_pred.max())
    min_val = min(y_real.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2)

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Make sure that 'conm' is the column with the company names
        # Adjust 'conm' to your column name if necessary
        sel.annotation.set(text=df_copy['conm'].iloc[sel.target.index],
                           position=(20, -20))  # Adjust position as required
        sel.annotation.get_bbox_patch().set(fc="white", alpha=0.6)

    plt.grid(True)
    plt.show()


def evaluate_model_performance(model, X, y):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy.stats import zscore, linregress
    import seaborn as sns
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white

    if not 'const' in X.columns:
        X = sm.add_constant(X)

    y_pred = model.predict(X)

    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {adjusted_r_squared}")

    residuals = y - y_pred
    assert len(y_pred) == len(y), "y_pred and y must have the same length"

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color="red")
    plt.title('Residuals diagram')
    plt.xlabel('Predicted value')
    plt.ylabel('Residues')
    plt.show()

    sm.qqplot(residuals, line='s')
    plt.title('Q-Q plot of residuals')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Distribution of residues')
    plt.xlabel('Résidues')
    plt.show()

    # Breusch-Pagan test for heteroskedasticity
    bp_test = het_breuschpagan(residuals, X)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print("Breusch-Pagan test for heteroscedasticity :", dict(zip(labels, bp_test)))

    # White's test for heteroscedasticity
    white_test = het_white(residuals, X)
    labels_white = ['Test Statistic', 'Test p-value', 'F-Statistic', 'F-Test p-value']
    print("White's test for heteroskedasticity :", dict(zip(labels_white, white_test)))
    dw_stat = durbin_watson(residuals)
    jb_stat, jb_pvalue = jarque_bera(residuals)
    condition_number = np.linalg.cond(X.values)
    print(f"Durbin-Watson statistic: {dw_stat}")
    print(f"Jarque-Bera test: Statistics = {jb_stat}, p-value = {jb_pvalue}")
    print(f"Number of conditions: {condition_number}")
    
    # Tests on coefficients (significance) 
    print("\nSignificance of coefficients:")
    print(model.summary().tables[1])
    split_index = len(y) // 2
    model_first_half = sm.OLS(y[:split_index], X[:split_index]).fit()
    model_second_half = sm.OLS(y[split_index:], X[split_index:]).fit()
    print("\nR-squared comparison - First half vs. second half of the data:")
    print(f"First half: {model_first_half.rsquared}, Second half: {model_second_half.rsquared}")


def compare_models_for_firm_value(df, features_columns, target_column='firm_value', test_size=0.2, random_state=42):
    # Separation of features and target
    X = df[features_columns]
    y = df[target_column]

    # Imputation of missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Division into learning and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=test_size, random_state=random_state)

    # Standardisation of characteristics
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Dictionary of models to compare
    models = {
        "Régression Linéaire": LinearRegression(),
        "Régression Ridge": Ridge(alpha=1.0)
    }

    # Settings for GridSearchCV with the Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Display of the best parameters and addition of the optimised model to the dictionary
    print(f"Better parameters for the Random Forest: {grid_search.best_params_}")
    models['Forêt Aléatoire'] = grid_search.best_estimator_
    
    r2_scores = {}  # Pour stocker le R^2 de chaque modèle

    # Entraînement (si nécessaire) et évaluation des modèles
    for name, model in models.items():
        if name != 'Forêt Aléatoire':  # La Forêt Aléatoire a déjà été entraînée par GridSearchCV
            model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_scores[name] = r2  # Stocker le R^2 dans le dictionnaire
        print(f"{name} - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Tracé de l'importance des features pour la forêt aléatoire
    forest_model = models['Forêt Aléatoire']
    importances = forest_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Importance of Features with Random Forest")
    plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X.shape[1]), [features_columns[i] for i in indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    print(f"R² spécifique pour la Forêt Aléatoire: {r2_scores['Forêt Aléatoire']:.4f}")

    # Prédiction des valeurs avec la forêt aléatoire et création d'un graphique
    forest_model = models['Forêt Aléatoire']
    y_pred_forest = forest_model.predict(X_test_scaled)  # Prédictions avec la forêt aléatoire

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_forest, alpha=0.3, color='blue')  # Plot des prédictions vs valeurs réelles
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted values vs. actual values for the Aleator Forest')
    # Tracer la ligne y=x pour la référence
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
    plt.show()