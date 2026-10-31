# Demand forecasting — Retail ML Platform

This guide describes the technical contract and operation of the Retail ML Platform demand forecast. It covers the data rules, feature matrix, models, metrics, published tables, and known limitations. It is intended for people who run, maintain, or consume the forecast.

## Overview

The workflow transforms paid sales into monthly product-location series, builds leakage-safe features, evaluates models, and publishes forecasts for three months.

~~~text
fct_sales + dimensions
        ↓
int_monthly_product_location_demand
        ↓
int_forecast_feature_matrix
        ↓
temporal training and evaluation
        ↓
fct_forecast_model_metrics
        ↓
fct_demand_forecast + fct_forecast_series_publication
        ↓
SKU-location forecast with review policy

int_forecast_feature_matrix
        ↓
agg_location_demand_forecast
        ↓
location-level planning for stores and distribution centers
        ↓
rankings and logistics screening
~~~

## Source data and demand definition

The sales source is retail_analytics.default_marts.fct_sales. Supporting dimensions are dim_product, dim_location, and dim_customer in the same catalog and schema. The source contains order-item grain, order and product keys, location, sale date, status, quantity, and commercial amounts.

### Grain and target

The modeling grain is demand_month + product_variant_id + location_id. The target is the quantity sold in the month:

~~~text
demand_qty = sum(quantity_qty)
~~~

The forecast represents units, not revenue.

### Sales filter

Only paid orders contribute to the target:

~~~sql
where order_status = 'paid'
~~~

Draft, canceled, or unreconciled orders are excluded from observed demand.

### Cutoff date and forecast horizon

The run accepts reference_date in YYYY-MM-DD format. If it is empty, Python uses the operational date in America/Sao_Paulo. The cutoff is the last day of the closed month before the reference date. The three forecast horizons are the following three months.

| reference_date | Cutoff | Forecast horizon |
|---|---|---|
| 2026-09-13 | 2026-08-31 | September–November 2026 |
| 2026-10-13 | 2026-09-30 | October–December 2026 |

The job resolves this context once and passes the same reference to dbt and the notebooks. It never derives the cutoff from MAX(sale_datetime), because the source contains future-dated records. Staging retains those rows for traceability, but the cutoff is applied before monthly aggregation, feature construction, training, and evaluation:

~~~sql
where cast(sale_datetime as date) <= cutoff_date
~~~

Rows after the cutoff remain available in the source but cannot be treated as realized demand or used by the model.

### Returns

The first version uses gross paid quantity. An audit found 4,612 returned units, about 0.8% of the observed paid quantity. Returns are not deducted because each return must first be matched to the correct sale and period.

A future version may publish net_demand_qty. The current rule is:

~~~text
demand_qty = quantity sold on paid orders
~~~

## Monthly series

The dbt model is dbt/models/intermediate/int_monthly_product_location_demand.sql.

| Field | Description |
|---|---|
| demand_month | First day of the month |
| year_number, quarter_number, month_number | Calendar attributes |
| product_variant_id, product_key | Product identifiers |
| sku, product_name, category_name, brand_name | Product attributes |
| location_id, location_key | Location identifiers |
| location_name, location_type, city, state | Location attributes |
| demand_qty | Quantity sold in the month |
| gross_revenue_brl | Gross revenue |
| order_count | Distinct orders |
| item_line_count | Order-item lines |
| days_with_sales | Days with positive quantity |
| has_positive_demand | Whether demand is above zero |
| history_month_count | Months from first sale through the cutoff |
| positive_demand_month_count | Months with positive demand |
| is_regression_eligible | Regression eligibility |
| forecast_strategy | Selected strategy |

Months with no sales after a series' first sale are filled with zero. Product-location pairs with no paid sales do not form a historical series.

### Series eligibility

A series can use regression when it has at least six months of history and at least three positive-demand months. Otherwise, it uses a fallback baseline.

| Condition | is_regression_eligible | forecast_strategy |
|---|---|---|
| At least 6 history months and 3 positive months | true | regression |
| Below either threshold | false | baseline_fallback |

## Feature matrix

The model is dbt/models/intermediate/int_forecast_feature_matrix.sql.

| Group | Fields |
|---|---|
| Identification | demand_month, product_variant_id, location_id, location_type |
| Target | demand_qty |
| Lags | lag_1_qty, lag_2_qty, lag_3_qty |
| Rolling windows | rolling_3m_avg_qty, rolling_6m_avg_qty, rolling_3m_sum_qty |
| Calendar | month_number, quarter_number, year_number |
| Product | sku, product_name, category_name, brand_name |
| Location | location_name, location_type, city, state |
| Flags | has_positive_demand, is_regression_eligible, is_training_eligible |

Lags and rolling windows use only months before the target month. The target month and future months are never used to build features.

- lag_1_qty, lag_2_qty, and lag_3_qty are demand_qty from one, two, and three months earlier for the same product_variant_id + location_id.
- rolling_3m_avg_qty is the average demand_qty over the prior three months, populated only when all three months exist before the target.
- rolling_6m_avg_qty is the average over the prior six months, populated only when all six months exist.
- rolling_3m_sum_qty is the sum over the prior three months, also requiring three complete months.
- year_number, quarter_number, and month_number are calendar components of the target demand_month.

## Modeling and evaluation

The baselines use three- and six-month moving averages. Short series use baseline_ma3 as a fallback.

Evaluated models include linear or Ridge regression, Random Forest Regressor, and Gradient Boosting Regressor or an equivalent available in the environment.

Evaluation preserves temporal order. For the reference run with cutoff 2026-08-31:

| Split | Period |
|---|---|
| Training | Through 2026-05-31 |
| Evaluation | 2026-06-01 through 2026-08-31 |
| Forecast | September through November 2026 |

Later runs evaluate the three closed months ending at the cutoff and forecast the following three months.

Starting with regression_v2_recursive, the three-month evaluation mirrors operational inference: month 2 uses the month 1 forecast as history, and month 3 uses the month 1 and month 2 forecasts. Actual test values are used only for metrics. Comparisons include baseline_zero as a sanity check and the same month in the previous year as a seasonal reference. This comparison covers regression-eligible series, not series that use fallback in the complete forecast.

In run model_forecast_2026_08, zero had the lowest WAPE for the evaluated cohort. Gradient Boosted Trees was the best regression but did not beat zero or the seasonal reference:

| Model | WAPE | MAE | RMSE |
|---|---:|---:|---:|
| Zero | 100.000% | 0.754 | 2.416 |
| Seasonal | 176.734% | 1.333 | 3.192 |
| Gradient Boosted Trees | 176.788% | 1.333 | 2.298 |
| Random Forest | 198.654% | 1.498 | 2.310 |
| Ridge | 256.267% | 1.933 | 2.445 |
| MA3 | 265.456% | 2.002 | 3.195 |
| MA6 | 303.322% | 2.288 | 3.078 |

Regression was not promoted for this final forecast snapshot. Notebook 18_series_model_selection.py selects zero, MA3, MA6, or seasonal for each product-location across three temporal windows. A candidate replaces zero only if it wins at least two windows and reduces combined WAPE by at least two percentage points. Notebook 03_generate_forecast.py applies that decision to all three horizons.

The selected model is persisted with each forecast. Training and evaluation are never randomly shuffled.

## Metrics

Metric functions are in ml/demand_forecasting/src/metrics.py.

| Metric | Interpretation |
|---|---|
| MAE | Mean absolute error in demand units |
| RMSE | Penalizes larger absolute errors |
| MAPE | Mean percentage error for rows with nonzero actual values |
| WAPE | Sum of absolute errors divided by absolute actual demand |

MAPE and WAPE are percentages and can exceed 100%. High values indicate substantial error and require caution. The best result in a run identifies only the best among the evaluated models; it does not, by itself, prove accuracy or reliability.

When every actual value is zero, MAPE and WAPE return NaN during calculation. Undefined values are stored as NULL.

Example:

~~~python
from ml.demand_forecasting.src.metrics import calculate_metrics

results = calculate_metrics(actual=[100, 200], predicted=[110, 180])
~~~

The result contains mae, rmse, mape, wape, and evaluated_rows. Metrics are stored in retail_analytics.default_marts.fct_forecast_model_metrics.

The grain is run_id + model_name + evaluation_window. run_id represents the logical model run, such as model_forecast_YYYY_MM, derived from the cutoff rather than a technical attempt. execution_id identifies the job execution and is recorded in parameters_json. Re-running a period updates the same logical snapshot instead of creating a duplicate.
## Published tables

~~~text
retail_analytics.default_marts.fct_forecast_model_metrics
retail_analytics.default_marts.fct_demand_forecast
retail_analytics.default_marts.fct_forecast_series_publication
retail_analytics.default_marts.agg_location_demand_forecast
retail_analytics.default_marts.agg_product_seasonality
retail_analytics.default_marts.agg_location_forecast_growth
retail_analytics.default_marts.agg_distribution_center_recommendation
retail_analytics.default_marts.dim_distribution_center_coverage
~~~

### Final SKU-location forecast

Notebook ml/demand_forecasting/notebooks/03_generate_forecast.py writes fct_demand_forecast.

The reference DEV run forecast_2026_08 produced:

| Item | Result |
|---|---:|
| Product-location series | 6,054 |
| Rows per horizon | 6,054 |
| Persisted rows | 18,162 |
| Unique keys | 18,162 |
| Negative forecasts | 0 |
| SKU-store series | 3,027 |
| SKU-store rows across all horizons | 9,081 |
| Policy | Per-series temporal winner |

For stores, 101 series selected a nonzero baseline: 27 use MA3, 12 use MA6, and 62 use the seasonal reference. The other 2,926 series remain at zero because no candidate showed a consistent gain across the three windows. There were 44 positive forecasts in September, 57 in October, and 56 in November 2026. Zero is a validated decision for intermittent demand, not a missing forecast.

Across the same three windows, the final per-series policy achieved combined WAPE of 98.606% and MAE of 1.624, compared with 100.000% WAPE and 1.647 MAE for zero. WAPE by window was 100.675%, 98.001%, and 98.381%. The aggregate gain is small and does not appear in the first window. The granular forecast therefore remains low-confidence and requires monthly review; it is not claimed to be highly accurate for each series.

The persistence key is run_id + forecast_month + product_variant_id + location_id. New runs use a stable snapshot_id in the format forecast_YYYY_MM. A retry for the same cutoff updates and reconciles the existing snapshot; another cutoff preserves a separate period.

Publication status, selected model, confidence, and per-series review requirement are stored in fct_forecast_series_publication at run_id + product_variant_id + location_id grain. This output satisfies the SKU-store forecast requirement but does not authorize automatic ordering. Individual decisions remain subject to policy and human review.

### Location-level aggregate for planning

Notebook 20_generate_location_aggregate_forecast.py publishes agg_location_demand_forecast at run_id + forecast_month + location_id grain. It uses demand at the same location in the same month of the previous year, with a recursive MA3 fallback when no annual match exists.

This is separate from the SKU-store forecast. It supports capacity planning, store and distribution center rankings, and logistics screening. It is not the sum of fct_demand_forecast and does not reconcile individual forecasts.

Reference snapshot forecast_2026_08 contains 18 rows: six locations for three months. Published monthly totals are 5,790, 8,934, and 12,061 units for September, October, and November 2026.

Manual notebook 21_location_forecast_retrospective_validation.py validates this output only after actuals become available. It does not train, recalculate, or modify the published forecast. It compares three monthly totals and 18 month-location combinations with paid sales after the cutoff, reporting absolute error and WAPE at both grains. Holdout actuals never feed training or publication of the evaluated snapshot.

For forecast_2026_08, the September–November 2026 holdout had 4,162 units of absolute error and 13.45% WAPE at monthly total grain. At the planning grain, the 18 month-location combinations had 4,998 units of absolute error and 16.15% WAPE. These are retrospective results for aggregated planning, not a training metric or a promise of SKU-store accuracy.

### Store growth

Notebook ml/demand_forecasting/notebooks/04_location_forecast_growth.py writes store rows to agg_location_forecast_growth.

It compares:

- actual_last_3m_demand_qty: realized demand in the previous three months;
- actual_start_month and actual_end_date: bounds of the realized period;
- forecast_next_3m_demand_qty: forecast over the next three horizons;
- forecast_start_month and forecast_end_month: forecast period bounds;
- growth_abs_qty: forecast demand minus realized demand;
- growth_pct: absolute growth divided by realized demand;
- growth_rank_abs and growth_rank_pct: ranking positions.

Series with fewer than three realized months receive growth_base_status=insufficient_actual_history. If realized demand is zero, the status is new_or_inactive_base and percentage values are NULL.

### Product seasonality

Notebook ml/demand_forecasting/notebooks/05_seasonality_ranking.py uses the monthly series and publishes agg_product_seasonality. The main analysis combines stores and warehouses at product_variant_id + month_number grain.

A product is eligible with at least 12 months of history and 6 months with positive sales. For each calendar month:

~~~text
seasonality_index = avg_monthly_demand_qty / overall_avg_monthly_demand_qty
~~~

The coefficient of variation across the 12 monthly averages ranks products:

~~~text
cv = stddev_pop(12 monthly averages) / overall average
~~~

Monthly labels are peak at index >= 1.20, low at index <= 0.80, and neutral between those thresholds.

Confidence uses the smallest number of observed years among the 12 months:

| Minimum observations | Confidence |
|---:|---|
| At least 3 | high |
| 2 | medium |
| Fewer than 2 | low |

The reference analysis selected product 608, SKU-462189, Outboard Motor 2640, with CV 0.908 and high confidence. Peak months were December through March; low months were May through November.

### Distribution center growth

The agg_location_forecast_growth table also stores rows with comparison_type=distribution_center.

Distribution center demand does not directly use warehouse location_id, because that key does not reliably represent the consumer's geographic origin. The analysis uses this proxy:

~~~text
dim_customer.state → macro-region → assigned distribution center
~~~

Current coverage:

| Macro-region | Assigned distribution center |
|---|---|
| North | location_id 3, AP |
| Northeast | location_id 4, PI |
| Central-West | location_id 2, PR |
| Southeast | location_id 2, PR |
| South | location_id 2, PR |

Rows with null or unknown states or no matching customer are excluded from regional aggregation and should be counted as a data quality metric.

In run forecast_2026_08, the Southern distribution center in PR had the largest absolute growth: 5,112.32 more forecast units than realized demand. Percentage growth is equal across centers because allocation preserves historical shares; use growth_abs_qty for the correct ranking.

Coverage evidence is stored in retail_analytics.default_marts.dim_distribution_center_coverage.

### Candidate regions for a new distribution center

agg_distribution_center_recommendation ranks macro-regions without their own distribution center by forecast_next_3m_demand_qty, the total forecast demand for the next three months. recommendation_score is that demand normalized by the largest candidate. growth_abs_qty is available for comparison with prior realized demand, but does not determine rank or score.

The screening card displays forecast_next_3m_demand_qty in units. It is not a score, absolute growth, or an opening decision. In reference snapshot forecast_2026_08, Central-West ranked first with about 4,005 units forecast over the three months and about 1,957 units of growth over the preceding realized period.

Central-West is the first monitoring priority among regions without a dedicated center. The current technical recommendation is not to open a center now: the observed increase is seasonal and does not prove structural growth. Reassess with a longer history, service levels, delivery times, inventory, and logistics costs.

## Distribution center location rules

Locations classified as store are used for point-of-sale analysis. Warehouse locations represent distribution centers and are not treated as direct consumer demand.

Order analysis found similar POS-channel shares for stores and warehouses. Because that is not operationally plausible, location_id is not treated as a reliable geographic attribution for distribution center growth.

Regional assignment uses states from the customer dimension. No geodesic distance is calculated; coverage is administrative and must be reviewed if the distribution center network changes.

## Module structure

~~~text
ml/
├── __init__.py
└── demand_forecasting/
    ├── notebooks/       Databricks execution and analysis
    ├── src/             configuration, metrics, and persistence
    ├── tests/           Python tests
    └── README.md        this guide
~~~

From the repository root, imports use the ml package:

~~~python
from ml.demand_forecasting.src.config import FORECAST_CONFIG, MARTS_SCHEMA, fq_table
~~~

Databricks Repos adds the repository root to the Python path. For local runs, start from the root or set PYTHONPATH=.; pytest uses the root path configured in pyproject.toml.
## Execution

### Prerequisites

These intermediate tables must be current:

~~~text
retail_analytics.default_intermediate.int_monthly_product_location_demand
retail_analytics.default_intermediate.int_forecast_feature_matrix
~~~

Model evaluation must be persisted so final forecast generation can select the best available model.

### Databricks job

The retail_analytics_data_pipeline bundle job runs the full sequence after dbt_transformation:

1. 02_train_evaluate_models.py evaluates baselines and regressions.
2. 16_segmented_baseline_backtest.py validates segment-level references.
3. 18_series_model_selection.py selects the temporal winner for each series.
4. 05_seasonality_ranking.py measures seasonality by product.
5. 15_series_policy.py records each series' policy and confidence.
6. 03_generate_forecast.py generates three horizons and MERGEs fct_demand_forecast.
7. 17_replenishment_planning_policy.py records the operational policy.
8. 20_generate_location_aggregate_forecast.py publishes the seasonal aggregate used by rankings.
9. 04_location_forecast_growth.py calculates store/POS growth.
10. 06_cd_forecast_growth.py calculates distribution center growth.
11. 07_distribution_center_recommendation.py ranks candidate regions.

Notebook 21_location_forecast_retrospective_validation.py is manual and read-only. It records retrospective evidence for aggregated forecasts after forecast months become actuals. It is not part of the job and does not change published tables.

Notebook 01_baseline.py remains available for manual audit but is not a job dependency. The training notebook already computes and persists both baseline metrics on the same test set as the regressions.

The store and distribution center growth notebooks MERGE into the same table with different comparison_type values. They run sequentially to avoid concurrent Delta write conflicts.

The job has two operational parameters:

- reference_date: empty uses the current date in America/Sao_Paulo; provide YYYY-MM-DD for reruns or backfills.
- execution_id: populated automatically with job.run_id, it identifies the attempt without changing the snapshot's logical identity.

Training fingerprints the feature matrix through the cutoff. If the period and fingerprint already have complete metrics, it reuses the model run; the generator also reuses a complete forecast. If late-arriving data changes the fingerprint, the same snapshot is recalculated and updated.

For a backfill, run the job with reference_date set to the desired operational date. For example, 2026-09-13 reconstructs forecast_2026_08 and publishes September through November. Later tasks receive model_run_id and snapshot_id through task values instead of selecting the latest record by created_at.

### Validate the forecast

~~~sql
SELECT
  forecast_month,
  COUNT(*) AS total_rows,
  COUNT(DISTINCT CONCAT_WS('|', product_variant_id, location_id)) AS series
FROM retail_analytics.default_marts.fct_demand_forecast
WHERE run_id = 'forecast_YYYY_MM'
GROUP BY forecast_month
ORDER BY forecast_month;
~~~

Check for negative forecasts:

~~~sql
SELECT COUNT(*) AS negative_rows
FROM retail_analytics.default_marts.fct_demand_forecast
WHERE run_id = 'forecast_YYYY_MM'
  AND forecast_demand_qty < 0;
~~~

Check for duplicate keys:

~~~sql
SELECT forecast_month, product_variant_id, location_id, COUNT(*) AS total
FROM retail_analytics.default_marts.fct_demand_forecast
WHERE run_id = 'forecast_YYYY_MM'
GROUP BY forecast_month, product_variant_id, location_id
HAVING COUNT(*) > 1;
~~~

Expected results are zero negative forecasts and no duplicate keys.

### Validate store growth

~~~sql
SELECT
  location_name,
  location_type,
  actual_last_3m_demand_qty,
  forecast_next_3m_demand_qty,
  growth_abs_qty,
  growth_pct,
  growth_rank_abs
FROM retail_analytics.default_marts.agg_location_forecast_growth
WHERE comparison_type = 'store'
ORDER BY growth_abs_qty DESC;
~~~

### Manual error audit by segment

Notebook 08_audit_model_errors.py analyzes the best regression on the holdout without changing the pipeline or writing tables. Run it manually in Databricks with the same code version as the run and compute compatible with Spark ML. Set the model_run_id widget, for example, to model_forecast_2026_08.

Before training only the winning model, the notebook compares the feature-matrix fingerprint, hyperparameters, and train/test sizes with the published run. It then checks that recalculated MAE, RMSE, MAPE, and WAPE are close to the stored values. If any check fails, the audit stops without presenting segment conclusions.

For regression_v1 runs, reproduction uses the original one-step evaluation. For regression_v2_recursive, it uses all three recursive steps.

Outputs show error by location, realized-demand band, and their combination. When actual demand is zero, WAPE is undefined; use absolute_error and error_share_pct to measure the contribution to total error. The notebook retrains and consumes compute, but does not write tables, publish models, or run the full job. It avoids cache/persist so it also works on Serverless compute.

Notebook 09_recursive_backtest.py tests the newer procedure before the job runs. It is read-only, accepts a historical cutoff such as 2025-11-30 to evaluate September–November 2025, and compares zero, MA3, MA6, and seasonal baselines with one or all regressions. It reports monthly and location totals and uses forecasts from prior steps rather than actual test values. It covers regression-eligible series and consumes compute, but does not deploy.

This backtest also offers gradient_boosted_trees_l1, an experimental candidate for intermittent demand. Unlike the operational GBT, it uses absolute loss, which aligns more closely with WAPE. It checks whether the candidate reduces small positive forecasts spread across many rows without sales. Output includes the median, percentiles, dispersion, and number of zero forecasts. This candidate is not used by notebooks 02 and 03 and must not be published based on one test. Run it on the latest cutoff and then on 2025-11-30; compare WAPE, MAE, and forecast distribution with zero and the other references. If it does not beat zero and the best reference at both cutoffs, review the grain or forecasting strategy before changing the operational job.

Because the L1 variant remained above the zero reference at cutoffs 2026-08-31 and 2025-11-30, notebook 09 also offers two_stage_gbt. Select model_name and run it separately with cutoff_date=2026-08-31 and cutoff_date=2025-11-30. It trains a binary classifier on all months to estimate sale occurrence and a regressor on positive-demand months to estimate quantity. The forecast is zero below the occurrence threshold and the conditional quantity above it. Thresholds 0.10, 0.25, 0.50, and 0.75 share one training run, but each uses its own recursive forecast history. Test actuals are used only for evaluation. Output includes false positives, missed sales, and mean probability by band. These probabilities are decision scores and are not necessarily calibrated. The candidate remains outside the job; do not select a production threshold from the best WAPE on one holdout.

### Sale-occurrence signal audit

When recursive models do not distinguish sale months from months without sales, run notebook 10_audit_occurrence_signal.py manually in Databricks. Its cutoff_dates widget defaults to 2025-11-30,2026-08-31 and runs both windows. It does not train models, write tables, or join the job. It reports sale rate by recent sale-month count, recency of the last sale, sales in the same month of the prior year, and category/location. It also compares first-test-month lags and rolling averages with history before the cutoff. Actuals from the three test months are used only to calculate displayed rates and volumes.

Interpret lift_vs_window as a group's sale rate divided by the rate for the full window. Values close to 1 across groups suggest these features have little signal for locating product-location sales. Differences in *_mismatches columns instead indicate a feature-matrix issue.

### Exploratory hierarchical backtest

Run notebook 11_hierarchical_backtest.py manually with cutoff_dates=2025-11-30,2026-08-31. It compares fixed forecasts by category and location: the same month in the previous year, capped annual trend, and the prior three-month average. It allocates totals to products with uniform weights, local demand over the prior 12 months, or product popularity across locations. It uses only data before each holdout, does not train Spark ML, write tables, or change the operational job.

The main table reports WAPE at product-location, category-location, location, and monthly-total grains on the same training-eligible series. A second table separates error by location type (store or distribution center). The zero comparator is a reference (100% WAPE when demand is positive). Better aggregate error alone does not satisfy the requirement to forecast each product at each store. Do not publish a candidate without improving required-grain error at both cutoffs and checking business constraints.

### Retrospective inventory availability audit

Manual notebook 12_audit_inventory_availability.py checks whether no sales during a holdout may coincide with an out-of-stock condition. It uses cutoffs 2025-11-30 and 2026-08-31 and reconstructs monthly balances for diagnosis from stg_stock_levels and signed stg_stock_movements. It reports snapshot coverage, inconsistent negative balances, sales with nonpositive reconstructed balances, and the share of no-sale months with possible unavailability.

This reconstruction does not prove lost demand and does not authorize using current inventory as a historical feature. Negative balances, incomplete movements, or low coverage make the result inconclusive. No table or forecast is changed. Revisit the training target only after validating historical inventory integrity and repeating the product-location backtest.

### Intermittent sales backtest

Manual notebook 14_intermittent_bayes_backtest.py tests an alternative compatible with available data without assuming that zero sales means out of stock. It separates the chance of a sale from positive quantity and combines local history with pre-cutoff signals for the same product, category/location, and calendar month. It also compares zero, a frozen moving average, and the same month in the prior year.

Candidate selective_hurdle_nested selects its threshold only from the three historical windows before the holdout and only for store series. Distribution centers do not contribute to occurrence probability or conditional quantity estimates. Thresholds are within [0,1]; always forecasting zero is the zero baseline, not an artificial threshold above 1. Selection uses F2 to emphasize sale detection and reports precision, recall, F2, average precision, and ROC AUC. Calibration WAPE and MAE are diagnostics; promotion still depends on final WAPE at both cutoffs and every month. The recall weight in F2 is exploratory until the business provides stockout and excess-inventory costs. Above the threshold, the selected output uses conditional sale quantity, evaluated with its own WAPE and MAE only on rows with realized sales.

In the store-only reference run, occurrence had insufficient discrimination: ROC AUC was 0.554 on the 2025 holdout and 0.511 on the 2026 holdout. Conditional quantity had WAPE of 49.90% and 50.96%, respectively, but the selective combination raised final WAPE to 371.03% and 617.34%. The median mixture remained near the zero baseline, with WAPE of 100.00% and 100.09%. This experiment shows that regression can estimate quantity when a sale occurs, but does not support replacing the per-series temporal policy in the published forecast.

Run with cutoff_dates=2025-11-30,2026-08-31. The main result is store_all/sku_location_wape. Also review cohort coverage, monthly WAPE, volume bias, positive forecasts, false positives, and missed sales. The automatic floor requires the candidate to beat all references at both cutoffs and every month; this is not an approval. The candidate is read-only, outside the job, and must not be published without positive backtest evidence and business review. With current sources, the metric represents observed paid sales, not latent demand.

If no Python cluster is available, open the SQL Editor, select the Serverless Starter Warehouse, and run all of sql/14_intermittent_bayes_backtest.sql. The query includes both cutoffs and produces the same candidate and reference families without creating or changing tables. passes_all_store_periods=true appears only when the candidate beats the best reference overall and in every month of both holdouts. The final decision still requires review of bias, coverage, and occurrence errors.

### Historical price and SKU-store frequency experiment

dbt/models/intermediate/int_forecast_event_features.sql creates an experimental monthly matrix with the last transacted price before the target month and sale frequency over the prior three and twelve months. Current list price, current inventory balance, and current lead time are excluded. Unknown price remains NULL in the table; the notebook uses a missing-value flag and imputes zero only when building the model vector.

Run dbt build --select int_forecast_event_features, then run notebook 22_event_two_stage_backtest.py in Databricks. It trains a sale-occurrence classifier and a positive-quantity regressor. The monthly forecast is sale probability multiplied by quantity conditional on a sale. For holdout months 2 and 3, commercial signals remain frozen at the origin; only the calendar month changes. Training uses only months before the first month of each holdout.

Output reports WAPE and MAE at SKU-store grain for both cutoffs and each month, comparing the candidate with zero, MA3, and seasonal baselines on the same store series. It also reports probability, conditional quantity, and final forecast per SKU-store. The notebook does not publish the candidate; publication requires beating all references at both cutoffs and reviewing volume bias. Inventory movements and receipts are deferred until their coverage and temporal quality are validated.

### Tests and dbt

~~~bash
python -m pytest ml/demand_forecasting/tests
dbt build --select int_monthly_product_location_demand int_forecast_feature_matrix+
~~~

## Known limitations

- Future-dated records remain in the source and staging for traceability; only demand and forecast models apply the operational cutoff.
- Short series use a baseline and have less historical evidence.
- Returns are not deducted from gross demand.
- Distribution center forecasts use proportional allocation, not an independent model per macro-region.
- Distribution center coverage is administrative and does not use real distance.
- Marina may show a snapshot while published-table queries are not connected.
- Reference results are evidence from one run and do not replace refreshing the tables.

## Maintenance

Update this guide when the demand source or filter, cutoff or horizon, series grain or features, models, metrics, table names or contracts, execution or validation procedures, or result interpretation limitations change.
