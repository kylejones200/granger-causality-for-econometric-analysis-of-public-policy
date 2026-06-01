# Granger Causality for Econometric Policy Analysis

This project demonstrates Granger causality testing to analyze causal relationships between economic time series variables.

## Business context

Granger causality is a statistical method used to test whether one time series can predict another. It is widely used in econometrics and public policy analysis to identify relationships between variables over time. Unlike traditional causality, Granger causality does not imply a direct cause-and-effect relationship. Instead, it indicates that past values of one variable contain information useful for predicting future values of another.

Granger causality helps policymakers understand dynamic relationships between economic indicators, social variables, and policy outcomes. It answers public policy questions such as whether changes in interest rates influence inflation, if public sentiment predicts policy adoption, or if unemployment rates affect consumer spending.

Granger causality tests whether past values of one time series improve the prediction of another through lag selection, model estimation, and hypothesis testing. Lag selection determines the optimal number of time periods needed to accurately capture temporal dependencies. Two models are then estimated: the unrestricted model includes lagged values of both series, while the restricted model includes only the dependent variable's lagged values. An F-test compares these models, where rejecting the null hypothesis indicates a predictive relationship.

## Article

Medium article: [Granger Causality for Econometric Analysis](https://medium.com/@kylejones_47003/granger-causality-for-econometric-analysis-of-public-policy-95d748643609)

## Project Structure

```
.
├── README.md           # This file
├── main.py            # Main entry point
├── config.yaml        # Configuration file
├── requirements.txt   # Python dependencies
├── src/               # Core functions
│   ├── core.py        # Granger causality functions
│   └── plotting.py    # Tufte-style plotting utilities
├── tests/             # Unit tests
├── data/              # Data files
└── images/            # Generated plots and figures
```

## Configuration

Edit `config.yaml` to customize:
- Data source (FRED series codes)
- Date ranges
- Granger causality test parameters (maxlag)
- Test directions and hypotheses

## Granger Causality

Granger causality tests whether past values of one variable help predict another variable beyond what past values of the second variable can predict. Key steps:
1. Test for stationarity (ADF test)
2. Apply differencing if needed
3. Run Granger causality tests in both directions
4. Interpret results (p-values < 0.05 suggest Granger causality)

## Caveats

- Data is fetched from FRED by default. Ensure internet connection.
- Both series must be stationary for valid Granger causality tests.
- Granger causality does not imply true causality, only predictive causality.
- Results depend on lag selection (maxlag parameter).

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).