import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def read_csv_and_extract_columns(file_path):
    try:
        df = pd.read_csv(file_path)
        return df['Adj Close'].values  # Extract the 'Adj Close' column as a one-dimensional array
    except Exception as e:
        print("Error:", e)
        return None

# Function to calculate log returns
def calculate_log_returns(prices):
    return np.diff(np.log(prices))

# Read data and calculate log returns
data_files = [
    'IBM.csv', 'AAPL.csv', 'IXIC.csv', 'MS.csv', 'TGT.csv',
    'RUT.csv', 'NYCB.csv', 'CAL.csv', 'GS.csv', 'BCS.csv'
    
]

log_returns = []

for file in data_files:
    prices = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\\' + file)
    if prices is not None:
        log_returns.append(calculate_log_returns(prices))

log_returns = np.array(log_returns)

# Calculate mean returns and covariance matrix
mean_returns = np.mean(log_returns, axis=1)
cov_matrix = np.cov(log_returns)

# Function to compute portfolio return and volatility
def portfolio_return(weights, returns):
    return np.sum(weights * returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Function to minimize portfolio volatility
def minimize_volatility(weights, cov_matrix):
    return portfolio_volatility(weights, cov_matrix)

# Generate random portfolios
num_assets = len(mean_returns)
num_portfolios = 10000
portfolios = []

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    portfolio_return_i = portfolio_return(weights, mean_returns)
    portfolio_volatility_i = portfolio_volatility(weights, cov_matrix)
    portfolios.append([portfolio_return_i, portfolio_volatility_i])

portfolios = np.array(portfolios)

# Optimize for the minimum volatility portfolio
initial_weights = np.ones(num_assets) / num_assets
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
optimal_portfolio = minimize(minimize_volatility, initial_weights, args=(cov_matrix,),
                              method='SLSQP', constraints=constraints)

# Efficient frontier points
target_returns = np.linspace(portfolios[:, 0].min(), portfolios[:, 0].max(), num=100)
efficient_portfolios = []

for target_return in target_returns:
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, mean_returns) - target_return},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = minimize(minimize_volatility, initial_weights, args=(cov_matrix,),
                      method='SLSQP', constraints=constraints)
    efficient_portfolios.append([target_return, result.fun])

efficient_portfolios = np.array(efficient_portfolios)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(portfolios[:, 1], portfolios[:, 0], c=portfolios[:, 0] / portfolios[:, 1], marker='o', cmap='viridis')
plt.plot(efficient_portfolios[:, 1], efficient_portfolios[:, 0], linestyle='-', color='r', label='Efficient Frontier')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier (Markowitz Bullet)')
plt.colorbar(label='Sharpe Ratio')
plt.legend()

# Function to compute Sharpe ratio
def sharpe_ratio(weights, returns, risk_free_rate):
    portfolio_return = np.sum(weights * returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_volatility

# Function to minimize negative Sharpe ratio (since we're maximizing Sharpe ratio)
def negative_sharpe_ratio(weights, returns, risk_free_rate):
    return -sharpe_ratio(weights, returns, risk_free_rate)

# Function to compute Sharpe ratio
def sharpe_ratio(weights, returns):
    portfolio_return = np.sum(weights * returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return / portfolio_volatility

# Function to minimize negative Sharpe ratio (since we're maximizing Sharpe ratio)
def negative_sharpe_ratio(weights, returns):
    return -sharpe_ratio(weights, returns)

# Optimize for the Tangency Portfolio (Maximum Sharpe Ratio Portfolio)
optimal_tangency_portfolio = minimize(negative_sharpe_ratio, initial_weights, args=(mean_returns,),
                                      method='SLSQP', constraints=constraints)

# Extract the optimal weights for the Tangency Portfolio
tangency_weights = optimal_tangency_portfolio.x
tangency_return= portfolio_return(tangency_weights, mean_returns)
tangency_risk= portfolio_volatility(tangency_weights, cov_matrix)
# Calculate the Sharpe ratio of the Tangency Portfolio
tangency_sharpe_ratio = sharpe_ratio(tangency_weights, mean_returns)

print("Portfolio Weights:\n  [IBM, AAPL,IXIC, MS, GS,TGT, RUT, NYCB, CAl,BCS ]\n", tangency_weights)
print("Portfolio Return:", tangency_return)
print("Portfolio Risk:", tangency_risk)
print("Portfolio Sharpe Ratio:", tangency_sharpe_ratio)
plt.plot(tangency_risk, tangency_return, color='r', marker="*")
plt.show()