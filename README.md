# Customer Lifetime Value Prediction

Predicts customer lifetime value (CLV) in USD for 50,000 e-commerce customers using random forest regression.

## Data

E-commerce customer dataset (50,000 rows, 53 columns) from Kaggle: https://www.kaggle.com/datasets/datascikhan/e-commerce-customer-segmentation-2026?select=E-commerce_Customer_Segmentation_2026.csv

## Approach

1. Explore the CLV distribution and how it varies by age and customer segment.
2. Remove leaky or derived columns: `total_spent_usd`, `customer_profitability_usd`, `monetary_score`, `rfm_score`, and `dataset_year`.
3. Split the data 80/20 with a fixed random seed (42).
4. Impute missing values with training-set medians, then scale the features.
5. Train a RandomForestRegressor with 100 trees.

## Results

| Set  | RMSE     | R²    |
|------|----------|-------|
| Train | $5,172  | 0.990 |
| Test  | $13,862 | 0.930 |

Test RMSE is about 24% of average CLV ($57,722). The gap between training and test error suggests overfitting.

## Next steps

- Use cross-validation and tune tree depth and leaf size.
- Compare against a gradient boosting model.
- Review feature importance to see which behaviours drive CLV.

## Running the project

```bash
pip install -r requirements.txt
```

Open the notebook and run all cells. It needs pandas, numpy, scikit-learn, matplotlib, and seaborn.
