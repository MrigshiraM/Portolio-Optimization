import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
np.random.seed(6)

def random_weights(n):
    weights= np.random.rand(n)
    weights /= np.sum(weights)
  
    return weights
def portfolio_volatility(weights, cov_matrix, b):

    total_budget = np.sum(weights)
    adjusted_weights = (weights / total_budget) 
  
    portfolio_variance = np.dot(adjusted_weights.T, np.dot(cov_matrix, adjusted_weights))*b
      
    portfolio_risk = np.sqrt(portfolio_variance)
    return portfolio_risk


def portfolio_return(weights, expected_returns, b):
  
    total_budget = np.sum(weights)
    adjusted_weights = (weights / total_budget) 
    
    
    portfolio_return = np.dot(adjusted_weights, expected_returns)* b
    return portfolio_return

def SA_PO(returns, cov_matrix, i_temp, f_temp, c_rate, max_iter,b):
    start_time = time.time()
    num_assets= len(returns)
    c_weights= random_weights(num_assets)
    print ("RANDOM",c_weights)
    c_returns= portfolio_return(c_weights, returns,b)
    c_volatility= portfolio_volatility(c_weights, cov_matrix,b)
    o_weights= c_weights
    o_returns= c_returns
    o_volatility= c_volatility
    temp=i_temp
        
    all_weights = [c_weights]
    all_returns = [c_returns]
    all_volatilities = [c_volatility]
    p=[0]
    
    for _ in range (max_iter):
        if temp< f_temp:
            break
        next_weights= c_weights + np.random.normal(scale=0.05,size=num_assets)
        
        next_weights /= np.sum(next_weights)
       
        next_returns= portfolio_return(next_weights, returns,b)
        next_volatility= portfolio_volatility(next_weights, cov_matrix,b)
        nc_weights= -c_weights
        nnext_weights= -next_weights
    
        nc_sum = np.sum(nc_weights)
    
        nnext_sum = np.sum(nnext_weights)
        
        
        if nnext_sum > nc_sum or random.random() < np.exp((nc_sum - nnext_sum)/temp):
            c_weights=next_weights
            c_returns= next_returns
            c_volatility= next_volatility         
            
        if c_returns > o_returns and c_volatility< o_volatility:
            o_weights= c_weights
            o_returns= c_returns
            o_volatility= c_volatility
            
        temp *= c_rate
                # Append current solution to lists
        all_weights.append(c_weights)
        all_returns.append(c_returns)     
        all_volatilities.append(c_volatility)
        p.append(c_weights[1])
        
    
   
    all_weights = np.array(all_weights)
    all_returns = np.array(all_returns)
    all_volatilities = np.array(all_volatilities)
      
    plt.figure(figsize=(10, 6))
    plt.plot( all_volatilities,all_returns, marker='o', linestyle='-')
    plt.plot(o_volatility, o_returns, color='red', marker='*')
    plt.ylabel('Returns')
    plt.xlabel('Volatility')
    plt.title('Movement of Points in Solution Space')
    plt.grid(True)
    plt.show()
    
    end_time = time.time()  # Record the end time
    runtime = end_time - start_time
    print("Runtime:", runtime, "seconds")
      
    return o_weights, o_returns, o_volatility


def read_csv_and_extract_columns(file_path):
    try:
        df = pd.read_csv(file_path)
        return df['Adj Close'].values  # Extract the 'Adj Close' column as a one-dimensional array
    except Exception as e:
        print("Error:", e)
        return None

b = 1

dataibm = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\IBM.csv')
dataappl = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\AAPL.csv')
dataixic = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\IXIC.csv')
datams = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\MS.csv')
datatgt = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\TGT.csv')
datarut = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\RUT.csv')
datanycb = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\NYCB.csv')
datacal = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\CAL.csv')
datags = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\GS.csv')
databcs = read_csv_and_extract_columns(r'C:\Users\mrigs\Desktop\portfolio_assets\BCS.csv')


def calculate_log_returns(prices):
    """
    Calculate the logarithmic returns from the adjusted close prices.
    """
    # Compute the percentage change in prices
    log_returns = np.diff(np.log(prices))  # Corrected calculation of log returns
    return log_returns

# Example usage:

log_returns_ibm = calculate_log_returns(dataibm)
log_returns_appl = calculate_log_returns(dataappl)
log_returns_ixic = calculate_log_returns(dataixic)
log_returns_ms = calculate_log_returns(datams)
log_returns_tgt = calculate_log_returns(datatgt)
log_returns_rut = calculate_log_returns(datarut)
log_returns_nycb = calculate_log_returns(datanycb)
log_returns_cal = calculate_log_returns(datacal)
log_returns_gs = calculate_log_returns(datags)
log_returns_bcs = calculate_log_returns(databcs)

# Create a DataFrame to store the log returns
log_returns_df = pd.DataFrame({
    'IBM': log_returns_ibm,
    'AAPL': log_returns_appl,
    'IXIC': log_returns_ixic,
    'MS': log_returns_ms,
    'TGT': log_returns_tgt,
    'RUT': log_returns_rut,
    'NYCB': log_returns_nycb,
    'CAL': log_returns_cal,
    'GS': log_returns_gs,
    'BCS': log_returns_bcs
})


# Calculate the cumulative sum of log returns to get the time series of returns

returns_series = log_returns_df.cumsum()

total_return=returns_series.iloc[-1]

# Plot the returns over time
returns_series.plot(figsize=(10, 6))
plt.title('Time Series of Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.show()

ex_returnibm = np.mean(log_returns_ibm)
ex_returnappl = np.mean(log_returns_appl)
ex_returnixic = np.mean(log_returns_ixic)
ex_returnms = np.mean(log_returns_ms)
ex_returngs = np.mean(log_returns_gs)
ex_returntgt = np.mean(log_returns_tgt)
ex_returnrut= np.mean(log_returns_rut)
ex_returnnycb = np.mean(log_returns_nycb)
ex_returncal = np.mean(log_returns_cal)
ex_returnbcs = np.mean(log_returns_bcs)

ex_returns = np.array([ex_returnibm, ex_returnappl,  ex_returnixic,ex_returnms,ex_returngs, ex_returntgt,
                       ex_returnrut,ex_returnnycb,ex_returncal,ex_returnbcs])



#print("dataibm:", (dataibm))
#print("dataappl:",(dataappl))
#print("dataixic:",(dataixic))

ibm_prices = log_returns_ibm
appl_prices = log_returns_appl
ixic_prices = log_returns_ixic
ms_prices= log_returns_ms
gs_prices= log_returns_gs
rut_prices = log_returns_rut
cal_prices = log_returns_cal
nycb_prices = log_returns_nycb
tgt_prices= log_returns_tgt
bcs_prices= log_returns_bcs




prices_matrix = np.array([ibm_prices, appl_prices, ixic_prices,ms_prices,gs_prices,tgt_prices,rut_prices,nycb_prices,cal_prices,bcs_prices ])


cov_matrix = np.cov(prices_matrix)

w, ret, risk=SA_PO(ex_returns, cov_matrix, 400, 0, 0.95 , 500, b)

print ('Portfolio Allocation: [IBM, AAPL,IXIC, MS, GS,TGT, RUT, NYCB, CAl,BCS ] \n ', w)
print('Portfolio Return: ', ret)
print('Portfolio Risk: ', risk)