# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 18:20:50 2024

@author: mrigs
"""

from dwave.system import DWaveSampler, EmbeddingComposite
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import dwave
import dimod.binary

t1=0.3
t2= 0.5
t3= 0.2
b=100
p=1/(2^(3))

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
    
    
]

log_returns = []

for file in data_files:
    prices = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\\' + file)
    if prices is not None:
        log_returns.append(calculate_log_returns(prices))

log_returns = np.array(log_returns)
log_returns_df = pd.DataFrame({
    'IBM': log_returns[0],
    'AAPL': log_returns[1],
    'IXIC': log_returns[2],
    'MS': log_returns[3],
    'TGT': log_returns[4],
  
})


# Calculate the cumulative sum of log returns to get the time series of returns

returns_series = log_returns_df.cumsum()

returns_series.plot(figsize=(10, 6))
plt.title('Time Series of Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()
# Calculate mean returns and covariance matrix
ex_returns = np.mean(log_returns, axis=1)
cov_matrix = np.cov(log_returns)



def calculate_portfolio_risk(weights, cov_matrix, b):
  
    total_budget = abs(np.sum(weights))
    adjusted_weights = (weights / total_budget) 
    
    portfolio_variance = np.dot(adjusted_weights.T, np.dot(cov_matrix, adjusted_weights))*b
    
    portfolio_risk = np.sqrt(portfolio_variance)
    return portfolio_risk


def calculate_portfolio_return(weights, expected_returns, b):
 
    total_budget = abs(np.sum(weights))
    adjusted_weights = (weights / total_budget) * b
    
    portfolio_return = np.dot(adjusted_weights, expected_returns)
    return portfolio_return


def portfolio_ising_model(expected_returns, covariance_matrix, t1, t2, t3, b, p):
    num_assets = len(expected_returns)
    quadratic_terms = {}
    linear_terms = {}

    for i in range(num_assets):
        for j in range(num_assets):
            quadratic_terms[(i, j)] = 1 / 4 * (t2 * b ** 2 * p ** 2 + t3 * covariance_matrix[i, j])
    
    sum_quadratic_terms = {}
    for i in range(num_assets):
        sum_quadratic_terms[i] = sum(quadratic_terms[i, j] for j in range(num_assets))



    for i in range(num_assets):
        linear_terms[i] = ((-t1 * expected_returns[i] - 2 * t2 * b ** 2 * p) / 2) + sum_quadratic_terms[i]
    total_linear_terms = sum(linear_terms.values())
    #bias= 1/4*((4*sum_quadratic_terms)) + 1/2*(total_linear_terms) + t2*b**2
    #linear_terms[i]+=bias
  
    h = {i: linear_terms[i] for i in range(num_assets)}
    J = {(i, j): quadratic_terms[i, j] for i in range(num_assets) for j in range(num_assets)}
  
    return h, J



def solve_portfolio_optimization(h, J):
    start_time = time.time()
    sampler = EmbeddingComposite(DWaveSampler())
    
    response = sampler.sample_ising(h, J, num_reads=200)
    end_time = time.time() 
    runtime = end_time - start_time
    print("Runtime:", runtime, "seconds")
    
    return response


def main():
   
     
    h, J = portfolio_ising_model(ex_returns, cov_matrix, t1,t2,t3,b,p)
    
    #  D-Wave quantum annealer
    response = solve_portfolio_optimization(h,J)
   
    #print('RESPONSE', response.first.sample)
   
    best_solution = response.first.sample
    weights = np.array([best_solution[i] for i in range(len(ex_returns))])  # Convert solution to NumPy array
    portfolio_risk = calculate_portfolio_risk(weights, cov_matrix,b)
    portfolio_return = calculate_portfolio_return(weights, ex_returns,b)
    

    print("Portfolio Allocation:\n [IBM, AAPL,IXIC, MS, TGT ] \n", weights)
    print("Portfolio Risk:", portfolio_risk)
    print("Portfolio Return:", portfolio_return)
    
    


if __name__ == "__main__":
    main()
    
