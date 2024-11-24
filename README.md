# Financial Data Analysis Tool

## Project Overview

This project is designed to analyze and optimize stock and cryptocurrency portfolios by leveraging various financial data analysis techniques. Built using Python, the tool integrates machine learning models like **Random Forest**, traditional statistical methods such as **Linear Regression**, and powerful tools like **Cross Validation** and **GridSearchCV** to ensure robust analysis and performance.

The primary goal of the project is to offer insights into the relationships between different financial metrics, assess risks, and optimize portfolio allocations. While the project is currently housed on the **test branch**, it is actively being refined, with plans to move it to the main branch after addressing some final improvements.

## Features

- **Data Retrieval**: 
  - The tool retrieves historical financial data for companies, including key financial indicators like revenue, stock price, debt ratios, and other fundamental financial metrics.
  - Data is fetched using **WRDS (Wharton Research Data Services)** and stored in a local Excel file for further analysis.

- **Data Preprocessing**: 
  - The project handles missing data with **SimpleImputer**, standardizes financial data using **StandardScaler**, and filters out irrelevant or incomplete data to ensure only reliable, clean data is used for modeling.

- **Random Forest Regressor**: 
  - A **Random Forest** model is used for predictive analysis to estimate key financial metrics and help in portfolio optimization. The model's performance is fine-tuned using **GridSearchCV** for optimal hyperparameter selection.

- **Linear Regression**: 
  - For comparison and more straightforward financial analysis, **Linear Regression** is implemented to explore linear relationships between financial variables.

- **Cross Validation**: 
  - **Cross Validation** techniques are used to evaluate model accuracy and reliability, ensuring that the models are not overfitted and can generalize well to unseen data.

- **Risk Assessment**: 
  - The project includes tools for assessing risk, such as the **Breusch-Pagan Test** for heteroscedasticity and **Durbin-Watson statistic** for autocorrelation in residuals.

- **Visualizations**: 
  - The tool produces a variety of **visualizations** using **matplotlib** and **seaborn**, including correlation heatmaps, boxplots, and scatter plots to help users better understand the data and model results.

- **Portfolio Optimization**: 
  - The tool aims to help users optimize their portfolios by providing predictive insights into key financial ratios and firm values, offering a clearer view of potential portfolio strategies.

## Installation

To use this tool, you'll need to set up your environment with the necessary dependencies. Follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-project-name.git
cd your-project-name
