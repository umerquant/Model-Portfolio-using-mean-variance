import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Load the data
data = pd.read_csv('portfolio_data.csv')

# Calculate the expected returns and covariance matrix
returns = data.mean()
cov_matrix = data.cov()

# Define the risk tolerance levels for the portfolios
risk_levels = [0.1, 0.2, 0.3]

# Define the optimization problem
def optimize_portfolio(risk_level):
    n_assets = len(data.columns)
    weights = cp.Variable(n_assets)
    ret = returns @ weights
    risk = cp.quad_form(weights, cov_matrix)
    objective = cp.Minimize(risk - risk_level * ret)
    constraints = [cp.sum(weights) == 1, weights >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return weights.value

# Generate the portfolios for each risk tolerance level
portfolios = []
for risk_level in risk_levels:
    weights = optimize_portfolio(risk_level)
    portfolios.append(weights)

# Plot the portfolios
plt.figure(figsize=(10, 6))
plt.plot(np.sqrt(np.diag(cov_matrix)), returns, 'o', markersize=10) 
plt.xlabel('Risk')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
colors = ['red', 'green', 'blue']
for i in range(len(portfolios)):
portfolio = portfolios[i]
plt.plot(np.sqrt(np.dot(portfolio.T, np.dot(cov_matrix, portfolio))), np.dot(portfolio.T, returns), 'o', color=colors[i], markersize=10, label=f'Risk Level: {risk_levels[i]}')
plt.legend()
plt.show()
