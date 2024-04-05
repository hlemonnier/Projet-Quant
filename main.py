from Calculation_and_retrievale import *

logging.basicConfig(filename='journal.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def main_analysis_pipeline():
    final_df = retrieve_data()
    df_with_ratio = ratio_calculation(final_df)
    df_filtered = filter_companies_with_data_for_2023(df_with_ratio)
    df_no_outliers = df_filtered.copy()
    df_no_outliers.to_excel('financial_data.xlsx', index=False)

    X, y = prepare_variables(df_no_outliers)
    X_reduced, vif = reduce_multicollinearity(X)
    print_vif(vif)
    selected_features, model_robust = perform_regression_analysis(X_reduced, y)
    print(model_robust.summary())
    columns_to_drop = [
        'total_debt', 'total_revenues', 'Mcap', 'equity', 'depreciation_amortization', 'ni', 'SD_StockPrice',
        'total_liabilities', 'long_term_debt', 'current_liabilities', 'current_assets', 'operating_income',
        'total_investments', 'txt', 'iid', 'tic', 'exchg', 'gvkey', 'datadate', 'conm', 'sic', 'gsector', 'gsubind',
        'shares_outstanding'
    ]
    df_for_analysis = df_no_outliers.drop(columns=columns_to_drop)
    # Calculating pairwise correlations
    correlation_matrix = df_for_analysis.corr()

    # Let's filter out the correlation matrix for high correlations excluding self-correlation
    high_correlation_pairs = correlation_matrix.unstack().sort_values(kind="quicksort", ascending=False)
    high_correlation_pairs = high_correlation_pairs[high_correlation_pairs < 1]  # remove self-correlation
    high_correlation_threshold = 0.7  # typically, a threshold of 0.75 or above is considered high
    high_correlation_pairs = high_correlation_pairs[abs(high_correlation_pairs) > high_correlation_threshold]

    print(high_correlation_pairs)

    X_for_prediction = sm.add_constant(X_reduced[selected_features])  # Correction ici
    y_pred = model_robust.predict(X_for_prediction)  # Utilisez X_for_prediction

    plot_predicted_vs_real(df_no_outliers, y, y_pred)  # Assurez-vous que cette fonction est définie correctement pour afficher le plot
    plot_residuals_vs_fitted(y_pred, y)
    evaluate_model_performance(model_robust, X_for_prediction, y)  # Corrigé pour utiliser X_for_prediction


def clean_outliers(df):
    df = df.copy()
    ratios_to_clean = [
        'roa', 'roe', 'current_ratio', 'debt_ratio', 'operating_margin',
        'firm_value', 'EBITDA', 'Mcap', 'E/D+E', 'EV/EBITDA', "EV/EBITDA(1+tr)",
        "SD_StockPrice", "EV/EBIT"
    ]
    return df


def prepare_variables(df, visualize_relationships=False):
    columns_to_drop = [
        'firm_value', 'stock_price', 'total_debt', 'total_revenues', 'Mcap',
        'equity', 'depreciation_amortization', 'ni', 'SD_StockPrice',
        'total_liabilities', 'long_term_debt', 'current_liabilities',
        'current_assets', 'operating_income', 'total_investments', 'txt',
        'iid', 'tic', 'exchg', 'gvkey', 'datadate', 'conm', 'sic', 'gsector',
        'gsubind', 'shares_outstanding'
    ]

    # Before dropping, check if visualization is requested
    if visualize_relationships:
        for column in df.drop(columns=columns_to_drop + ['firm_value']).columns:
            sns.scatterplot(x=df[column], y=df['firm_value'])
            plt.xlabel(column)
            plt.ylabel('Firm Value (log)')
            plt.title(f'Relationship between {column} and Firm Value')
            plt.show()
    # Constants for transformation
    small_constant = 1e-4

    # Apply square root transformation after adding a small constant
    df['current_ratio'] = np.sqrt(df['current_ratio'] + small_constant)
    df['debt_ratio'] = np.sqrt(df['debt_ratio'] + small_constant)
    df['E/D+E'] = np.sqrt(df['E/D+E'] + small_constant)
    df['EV/EBITDA(1+tr)'] = np.sqrt(df['EV/EBITDA(1+tr)'] + small_constant)

    # Proceed with data preparation
    for column in ['EBITDA', 'EV/EBIT']:
        df[column] = df[column].apply(lambda x: np.log(x) if x > 0 else 0)

    df['firm_value'] = df['firm_value'].apply(lambda x: np.log(x) if x > 0 else 0)
    X = df.drop(columns=columns_to_drop)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

    y = df['firm_value'].replace([np.inf, -np.inf], np.nan).fillna(0)

    assert len(X) == len(y), "X and y must have the same number of rows"

    return X, y


def print_vif(vif_data):
    print("VIF for each variable:\n", vif_data)
    print("\nVariables with VIF > 5 will be dropped to reduce multicollinearity.")


def reduce_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["features"] = X.columns

    # Calcul du VIF pour chaque variable
    vif_data["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Identifier les colonnes avec un VIF infini
    infinite_vif_columns = vif_data[vif_data["VIF Factor"] == np.inf]["features"].tolist()

    # Identifier les colonnes avec un VIF supérieur à 5 mais pas infini
    high_vif_columns = vif_data[(vif_data["VIF Factor"] > 5) & (vif_data["VIF Factor"] < np.inf)]["features"].tolist()

    # Combinez les deux listes des colonnes à exclure
    columns_to_exclude = list(set(infinite_vif_columns + high_vif_columns))

    # Exclure les colonnes avec un VIF élevé ou infini
    X_reduced = X.drop(columns=columns_to_exclude)

    return X_reduced, vif_data


def perform_regression_analysis(X, y, use_transformation=False, use_wls=False):
    """
    Performs regression analysis with options for transformation and Weighted Least Squares.

    Parameters:
    - X: DataFrame of predictors.
    - y: Series or array-like of the target variable.
    - use_transformation: Boolean indicating whether to apply log transformation.
    - use_wls: Boolean indicating whether to use Weighted Least Squares if heteroscedasticity is detected.

    Returns:
    - A tuple of selected features and the regression model.
    """

    # Optionally apply log transformation to predictors and target variable
    if use_transformation:
        X = np.log(X + 1)  # Adding 1 to handle zero values
        y = np.log(y + 1)

    # Stepwise selection might be a custom function you've defined or adapted
    selected_features = stepwise_selection(X, y, verbose=True)
    print("Selected features after stepwise selection:\n", selected_features)

    X_final = sm.add_constant(X[selected_features])

    # Fit initial OLS model
    initial_model = sm.OLS(y, X_final).fit()

    # Perform the Breusch-Pagan test
    bp_test_stat, bp_p_value, _, _ = het_breuschpagan(initial_model.resid, initial_model.model.exog)
    print('Breusch-Pagan Test Statistic:', bp_test_stat)
    print('p-value du test Breusch-Pagan:', bp_p_value)

    # Decide on using WLS or robust errors based on p-value and the use_wls flag
    if bp_p_value < 0.05 and use_wls:
        print("Significant heteroscedasticity detected; adjusting with WLS.")
        # Using the inverse of squared residuals as weights; adjust according to your needs
        weights = 1.0 / (initial_model.resid ** 2 + 1e-5)
        model_adjusted = sm.WLS(y, X_final, weights=weights).fit()
    elif bp_p_value < 0.05:
        print("Significant heteroscedasticity detected; adjusting with robust standard errors.")
        model_adjusted = initial_model.get_robustcov_results(cov_type='HC3')
    else:
        model_adjusted = initial_model

    return selected_features, model_adjusted


def plot_residuals_vs_fitted(y_pred, y):
    residuals = y - y_pred
    plt.scatter(y_pred, residuals)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.axhline(0, color='red', linestyle='dashed')
    plt.show()


if __name__ == '__main__':
    main_analysis_pipeline()
