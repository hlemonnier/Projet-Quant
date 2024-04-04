from Calculation_and_retrievale import *
logging.basicConfig(filename='journal.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')



# List of financial ratios to clean for outliers
ratios_to_clean = [ 'roa', 'roe', 'current_ratio', 'debt_ratio', 'operating_margin', 'firm_value', 'EBITDA', 'Mcap', 'E/D+E','EV/EBITDA',"EV/EBITDA(1+tr)","SD_StockPrice","EV/EBIT"]

# Retrieve initial data
final_df = retrieve_data()

# Apply filtering
df_filtered = filter_companies_with_data_for_2023(final_df)

# Calculate financial ratios for filtered companies
df_with_ratio = ratio_calculation(df_filtered)

# Copy the dataframe to preserve the original
df_no_outliers = df_with_ratio.copy()

# Remove outliers for each financial ratio
for ratio in ratios_to_clean:
    df_no_outliers = remove_outliers(df_no_outliers, 'sic', ratio)

# Exporting the final data without outliers to Excel
df_no_outliers.to_excel('financial_data.xlsx', index=False)

# Dropping specified columns to prepare explanatory variables X
columns_to_drop = [
    'iid', 'tic', 'exchg', 'gvkey', 'datadate', 'conm', 'sic', 'gsector', 'gsubind', 'shares_outstanding'
]
X = df_no_outliers.drop(columns=columns_to_drop)

# Preparing the target variable y (column 'roa')
y = df_no_outliers['firm_value']

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

# Effectuez le test de Breusch-Pagan sur les résidus
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_test_stat, bp_p_value, _, _ = bp_test

# Imprimez les résultats du test
print('Statistique de test Breusch-Pagan:', bp_test_stat)
print('p-value du test Breusch-Pagan:', bp_p_value)

# Prenez une décision basée sur la p-value
threshold = 0.05  # Définissez votre seuil ici
if bp_p_value < threshold:
    print(f"La p-value du test Breusch-Pagan est {bp_p_value}, indiquant une hétéroscédasticité significative.")
    # Ici, vous pouvez envisager des corrections pour l'hétéroscédasticité
    # Par exemple, utiliser des erreurs standards robustes dans votre modèle
    model_robust = sm.OLS(y, X_final).fit(cov_type='HC3')
    print("Modèle ajusté avec des erreurs standards robustes.")
else:
    print(f"La p-value du test Breusch-Pagan est {bp_p_value}, il n'y a pas de preuve significative d'hétéroscédasticité.")
    # Vous pouvez continuer avec votre modèle actuel
    print("Modèle ajusté sans besoin de corrections pour l'hétéroscédasticité.")
# Tracer les valeurs prédites par rapport aux valeurs réelles
y_pred = model_robust.predict(X_final)
print(model_robust.summary())
bp_test = het_breuschpagan(model_robust.resid, model_robust.model.exog)
bp_test_stat, bp_p_value, _, _ = bp_test
# Imprimez les résultats du test
print('Statistique de test Breusch-Pagan:', bp_test_stat)
print('p-value du test Breusch-Pagan:', bp_p_value)


# Call the function with the required arguments
plot_predicted_vs_real(df_with_ratio, y, y_pred)

# Tracer les résidus du modèle
residuals = y - y_pred

plt.scatter(y_pred, model_robust.resid)
plt.xlabel('Valeurs prédites')
plt.ylabel('Résidus')
plt.title('Residuals vs Fitted')
plt.axhline(0, color='red', linestyle='dashed')
plt.show()

def evaluate_model_performance(model_robust, X, y, y_pred, df):
    """
    Évalue la performance du modèle de régression en calculant et affichant le R² et le R² ajusté,
    et en traçant les graphiques des résidus et un Q-Q plot.
    
    Parameters:
    - model: Le modèle de régression ajusté.
    - X: Les variables explicatives utilisées pour ajuster le modèle.
    - y: La variable cible réelle.
    - y_pred: Les valeurs prédites par le modèle.
    - df: Le DataFrame contenant les données, utilisé pour afficher les noms des entreprises dans le graphique de dispersion.
    """
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    import seaborn as sns

    # Calcul du R² et du R² ajusté
    r_squared = model.rsquared
    adjusted_r_squared = model.rsquared_adj
    print(f"R-squared: {r_squared}")
    print(f"Adjusted R-squared: {adjusted_r_squared}")

    # Calcul des résidus
    residuals = y - y_pred

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
    bp_test = het_breuschpagan(residuals, model.model.exog)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    print("Test de Breusch-Pagan pour l'hétéroscédasticité :", dict(zip(labels, bp_test)))


# Après avoir construit le modèle et prédit les valeurs y_pred
evaluate_model_performance(model_robust, X_final, y, y_pred, df_with_ratio)

