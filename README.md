# House Price Prediction

Predicting house sale prices in Ames, Iowa using machine learning.

## Result
Final model (blend of Lasso + XGBoost + CatBoost) predicts prices with an average error of $13,043 (8.1%) and R² of 0.941 on unseen houses.

## Dataset
Ames Housing dataset: 1,460 houses, 81 columns (from OpenML).

## Steps
1. **Missing values:** dropped 4 columns with over 60% missing; filled "no feature" blanks with "None"; median for numbers, mode for text.
2. **Skewness:** price was right-skewed (1.88), fixed with log transform (0.12).
3. **Outliers:** removed 2 houses with huge size but very low price; capped LotArea, LotFrontage, MasVnrArea using IQR.
4. **Feature engineering:** created TotalSF, TotalBath, HouseAge, RemodAge, TotalPorch, HasGarage, HasBasement, HasFireplace. TotalSF became the strongest feature (correlation 0.83).
5. **Encoding:** label encoding for quality columns, one-hot encoding for the rest.
6. **Scaling:** StandardScaler (fit on train only).
7. **Models:** compared 10 models.
8. **Validation:** 5-fold cross-validation.
9. **Tuning:** RandomizedSearchCV on XGBoost.
10. **Ensemble:** averaged the top 3 models.

## Model Comparison (Test set)
| Model | R² | MAE | MAPE |
|---|---|---|---|
| Linear Regression | 0.805 | $17,561 | 10.3% |
| Ridge | 0.919 | $14,975 | 9.1% |
| Lasso | 0.925 | $14,339 | 8.7% |
| ElasticNet | 0.922 | $14,615 | 8.8% |
| Decision Tree | 0.685 | $26,791 | 16.5% |
| Random Forest | 0.900 | $16,390 | 10.3% |
| KNN | 0.758 | $24,527 | 14.7% |
| SVR | 0.793 | $20,264 | 12.7% |
| XGBoost | 0.934 | $14,058 | 8.7% |
| CatBoost | 0.941 | $13,103 | 8.2% |
| **Blend (final)** | **0.941** | **$13,043** | **8.1%** |

## Key Learnings
- Plain Linear Regression overfit (train 0.95 vs test 0.80). Regularisation (Lasso) fixed it.
- Lasso removed 61 useless features automatically.
- K-Fold showed the top 3 models are almost equal (0.909 to 0.915). One test split can be misleading.
- Tuning gave only a small gain. Feature engineering mattered more.

## Tools
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, XGBoost, CatBoost, joblib