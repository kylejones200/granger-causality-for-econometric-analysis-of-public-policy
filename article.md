# Granger Causality for Econometric Analysis of Public Policy Granger causality is a statistical method used to test whether one time
series can predict another. It is widely used in econometrics and...

### Granger Causality for Econometric Analysis of Public Policy
Granger causality is a statistical method used to test whether one time
series can predict another. It is widely used in econometrics and public
policy analysis to identify relationships between variables over time.
Unlike traditional causality, Granger causality does not imply a direct
cause-and-effect relationship. Instead, it indicates that past values of
one variable contain information useful for predicting future values of
another.

Granger causality helps policymakers understand dynamic relationships
between economic indicators, social variables, and policy outcomes. It
answers public policy questions such as whether changes in interest
rates influence inflation, if public sentiment predicts policy adoption,
or if unemployment rates affect consumer spending.

Granger causality tests whether past values of one time series improve
the prediction of another through lag selection, model estimation, and
hypothesis testing. Lag selection determines the optimal number of time
periods needed to accurately capture temporal dependencies. Two models
are then estimated: the unrestricted model includes lagged values of
both series, while the restricted model includes only the dependent
variable's lagged values. An F-test compares these models, where
rejecting the null hypothesis indicates a predictive relationship.

Granger causality is directional, testing whether variable X predicts Y
and vice versa, allowing identification of bidirectional relationships.
This chapter demonstrates both directions using Python to ensure
comprehensive policy analysis.

#### Assumptions and Limitations
Granger causality relies on stationarity, the absence of omitted
variable bias, and proper temporal ordering. Stationarity means that the
mean and variance of a time series do not change over time, which
prevents spurious results. The Augmented Dickey-Fuller (ADF) test helps
confirm stationarity, and differencing techniques can achieve it if
necessary. Avoiding omitted variable bias requires thorough domain
knowledge and exploratory data analysis, while temporal ordering assumes
that the cause precedes the effect, requiring accurate chronological
data.

This article explains these assumptions and demonstrates Granger
causality implementation in Python to ensure valid causal inference.

#### Case Study: Unemployment Rates and Consumer Spending
This case study evaluates the relationship between unemployment rates
and consumer spending. Economic theory suggests rising unemployment may
reduce consumer spending by decreasing confidence, thus slowing economic
growth. Conversely, declining consumer spending might lead to job
losses, increasing unemployment. Granger causality tests both
directions, illuminating the interplay between these variables.

The research examines whether changes in unemployment rates predict
changes in consumer spending and whether consumer spending predicts
unemployment rates. Monthly data from January 2010 to December 2022
includes the unemployment rate (percentage) and consumer spending
(billions of dollars).

#### Implementing Granger Causality in Python
The analysis begins by importing necessary libraries and loading the
dataset. I am getting the data from
[FRED](https://fred.stlouisfed.org/series/PCE).

```python
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Fetch data from FRED
start_date, end_date = '2010-01-01', '2022-12-31'
df = pd.concat([
    web.DataReader('UNRATE', 'fred', start_date, end_date),
    web.DataReader('PCE', 'fred', start_date, end_date)
], axis=1).rename(columns={'UNRATE': 'unemployment_rate', 'PCE': 'consumer_spending'})

# Reset index to use date column explicitly
df = df.reset_index().rename(columns={'DATE': 'date'})

# Save DataFrame to CSV
df.to_csv('unemployment_spending.csv', index=False)

# Plot data with proper years on x-axis
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title('Unemployment Rate and Consumer Spending Over Time')
ax1.set_xlabel('Year')
ax1.set_ylabel('Unemployment Rate (%)', color='red')
ax1.plot(df['date'], df['unemployment_rate'], color='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Consumer Spending (Billions)', color='blue')
ax2.plot(df['date'], df['consumer_spending'], color='blue')

plt.savefig('unemployment_consumer_spending.png')
plt.show()
```


<figcaption>The exploratory analysis visualizes how unemployment rates
and consumer spending evolve together over time. The unemployment rate
appears volatile with economic cycles, whereas consumer spending
consistently trends upward, reflecting economic growth.</figcaption>


#### Stationarity Tests
``` 
# Stationarity tests (ADF)
for col in ['unemployment_rate', 'consumer_spending']:
    adf_result = adfuller(df[col])
    print(f'{col} ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}')

# Differencing to achieve stationarity
df['unemployment_rate_diff'] = df['unemployment_rate'].diff()
df['consumer_spending_diff'] = df['consumer_spending'].diff()

# ADF Test after differencing
for col in ['unemployment_rate_diff', 'consumer_spending_diff']:
    adf_result = adfuller(df[col].dropna())
    print(f'{col} ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}')
```

Initial ADF tests indicated both unemployment and consumer spending were
non-stationary, with p-values of 0.079 and 0.997 respectively. After
applying first differences to each series, both became stationary,
confirmed by ADF tests with highly significant p-values near zero. This
step ensured valid causal analysis.

#### Granger Causality Test Results
``` 
# Granger causality tests
print('\nGranger Causality Tests:')
print('Does unemployment rate Granger-cause consumer spending?')
granger_test_ur_cs = grangercausalitytests(df[['consumer_spending_diff', 'unemployment_rate_diff']].dropna(), maxlag=4)

print('\nDoes consumer spending Granger-cause unemployment rate?')
granger_test_cs_ur = grangercausalitytests(df[['unemployment_rate_diff', 'consumer_spending_diff']].dropna(), maxlag=4)
```

Granger causality tests strongly indicated that unemployment predicts
consumer spending at lags of one to four months, with p-values near
zero, demonstrating robust predictive capability. Similarly, consumer
spending significantly predicts unemployment rates across the same lag
periods. These results confirm a clear bidirectional relationship
between the variables, aligning closely with economic theory: improved
employment stimulates consumption, and rising consumer spending
encourages job creation.

#### Policy Implications and Recommendations
These findings highlight a reciprocal relationship between unemployment
rates and consumer spending. Policymakers could leverage this dynamic,
using employment indicators to predict consumer economic behavior and
vice versa. For example, interventions designed to reduce unemployment,
such as job creation programs or training initiatives, might also
stimulate consumer spending. Conversely, policy measures aimed at
increasing consumer purchasing power --- such as stimulus payments or
tax incentives --- could indirectly boost employment by driving economic
demand.

Overall, Granger causality analysis offers policymakers valuable
predictive insights, helping them understand the dynamic relationships
fundamental to effective economic planning and intervention.
::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[March 22, 2025](https://medium.com/p/95d748643609).

[Canonical
link](https://medium.com/@kyle-t-jones/granger-causality-for-econometric-analysis-of-public-policy-95d748643609)

Exported from [Medium](https://medium.com) on November 10, 2025.
