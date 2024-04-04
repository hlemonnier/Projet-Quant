from Calculation_and_retrievale import *

logging.basicConfig(filename='journal.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


def main_analysis_pipeline():
    final_df = retrieve_data()
    df_filtered = filter_companies_with_data_for_2023(final_df)
    df_with_ratio = ratio_calculation(df_filtered)
    df_no_outliers = clean_outliers(df_with_ratio)
    df_no_outliers.to_excel('financial_data.xlsx', index=False)

    X, y = prepare_variables(df_no_outliers)
    X_reduced, vif = reduce_multicollinearity(X)
    print_vif(vif)
    selected_features, model_robust = perform_regression_analysis(X_reduced, y)

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
    for ratio in ratios_to_clean:
        df = remove_outliers(df, 'sic', ratio)
    return df


def prepare_variables(df):
    columns_to_drop = [
        'iid', 'tic', 'exchg', 'gvkey', 'datadate', 'conm', 'sic', 'gsector', 'gsubind', 'shares_outstanding'
    ]
    X = df.drop(columns=columns_to_drop)
    # Assurez-vous que toutes les valeurs sont numériques et gère les erreurs/NaN
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

    # Appliquer le logarithme sur 'firm_value', en s'assurant d'éviter les valeurs non positives
    df['firm_value'] = df['firm_value'].apply(lambda x: np.log(x) if x > 0 else 0)
    y = df['firm_value'].replace([np.inf, -np.inf], np.nan).fillna(0)

    assert len(X) == len(y), "X and y must have the same number of rows"
    return X, y


def print_vif(vif_data):
    print("VIF for each variable:\n", vif_data)
    print("\nVariables with VIF > 5 will be dropped to reduce multicollinearity.")


def reduce_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["features"] = X.columns
    vif_data["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    high_vif_columns = vif_data[vif_data["VIF Factor"] > 5]["features"].tolist()
    X_reduced = X.drop(columns=high_vif_columns)
    return X_reduced, vif_data


def perform_regression_analysis(X, y):
    selected_features = stepwise_selection(X, y, verbose=True)
    print("Selected features after stepwise selection:\n", selected_features)
    X_final = sm.add_constant(X[selected_features])
    initial_model = sm.OLS(y, X_final).fit()

    # Test de Breusch-Pagan
    bp_test = het_breuschpagan(initial_model.resid, initial_model.model.exog)
    bp_test_stat, bp_p_value, _, _ = bp_test
    print('Statistique de test Breusch-Pagan:', bp_test_stat)
    print('p-value du test Breusch-Pagan:', bp_p_value)

    # Ajustement avec des erreurs standards robustes si nécessaire
    if bp_p_value < 0.05:
        print("Hétéroscédasticité significative détectée; ajustement avec des erreurs standards robustes.")
        model_robust = sm.OLS(y, X_final).fit(cov_type='HC3')
    else:
        model_robust = initial_model

    return selected_features, model_robust


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
