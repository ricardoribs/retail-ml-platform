-- Read-only backtest for the Databricks SQL Editor.
-- Select the Serverless Starter Warehouse and run the entire query.
-- Evaluates Sep-Nov 2025 and Jun-Aug 2026 without using holdout actuals
-- to build the forecasts.

with cutoffs as (
    select
        cutoff_date,
        add_months(trunc(cutoff_date, 'month'), -2) as test_start,
        date_sub(add_months(trunc(cutoff_date, 'month'), -2), 1) as train_end
    from values
        (date '2025-11-30'),
        (date '2026-08-31') as selected_cutoffs(cutoff_date)
),

source_data as (
    select
        demand_month,
        product_variant_id,
        location_id,
        coalesce(category_name, '__missing__') as category_name,
        coalesce(location_type, '__missing__') as location_type,
        cast(demand_qty as double) as demand_qty
    from retail_analytics.default_intermediate.int_forecast_feature_matrix
    where demand_month <= date '2026-08-31'
),

eligible_series as (
    select
        c.cutoff_date,
        c.test_start,
        c.train_end,
        f.product_variant_id,
        f.location_id,
        max_by(f.category_name, f.demand_month) as category_name,
        max_by(f.location_type, f.demand_month) as location_type
    from cutoffs as c
    inner join source_data as f
        on c.train_end >= f.demand_month
    group by
        c.cutoff_date,
        c.test_start,
        c.train_end,
        f.product_variant_id,
        f.location_id
    having count(distinct f.demand_month) >= 6
        and sum(case when f.demand_qty > 0 then 1 else 0 end) >= 3
),

history as (
    select
        e.cutoff_date,
        e.test_start,
        e.train_end,
        e.product_variant_id,
        e.location_id,
        e.category_name,
        e.location_type,
        f.demand_month,
        f.demand_qty
    from eligible_series as e
    inner join source_data as f
        on e.product_variant_id = f.product_variant_id
        and e.location_id = f.location_id
        and add_months(e.test_start, -24) <= f.demand_month
        and e.train_end >= f.demand_month
),

local_stats as (
    select
        cutoff_date,
        test_start,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        sum(
            case
                when demand_month >= add_months(test_start, -12) then 1
                else 0
            end
        ) as recent_rows,
        sum(
            case
                when demand_month >= add_months(test_start, -12)
                    and demand_qty > 0 then 1
                else 0
            end
        ) as recent_positive_rows,
        sum(case when demand_qty > 0 then 1 else 0 end) as positive_rows,
        sum(case when demand_qty > 0 then demand_qty else 0 end) as positive_qty,
        sort_array(
            filter(
                collect_list(case when demand_qty > 0 then demand_qty end),
                value -> value is not null
            )
        ) as positive_values,
        sum(
            case
                when demand_month >= add_months(test_start, -3) then demand_qty
                else 0
            end
        ) / 3.0 as ma3_frozen
    from history
    group by
        cutoff_date,
        test_start,
        product_variant_id,
        location_id,
        category_name,
        location_type
),

product_stats as (
    select
        cutoff_date,
        test_start,
        product_variant_id,
        sum(
            case
                when demand_month >= add_months(test_start, -12) then 1
                else 0
            end
        ) as recent_rows,
        sum(
            case
                when demand_month >= add_months(test_start, -12)
                    and demand_qty > 0 then 1
                else 0
            end
        ) as recent_positive_rows,
        sum(case when demand_qty > 0 then 1 else 0 end) as positive_rows,
        sum(case when demand_qty > 0 then demand_qty else 0 end) as positive_qty,
        sort_array(
            filter(
                collect_list(case when demand_qty > 0 then demand_qty end),
                value -> value is not null
            )
        ) as positive_values
    from history
    group by cutoff_date, test_start, product_variant_id
),

group_stats as (
    select
        cutoff_date,
        test_start,
        location_id,
        category_name,
        count(*) as all_rows,
        sum(case when demand_qty > 0 then 1 else 0 end) as all_positive_rows,
        sum(
            case
                when demand_month >= add_months(test_start, -12) then 1
                else 0
            end
        ) as recent_rows,
        sum(
            case
                when demand_month >= add_months(test_start, -12)
                    and demand_qty > 0 then 1
                else 0
            end
        ) as recent_positive_rows,
        sum(case when demand_qty > 0 then 1 else 0 end) as positive_rows,
        sum(case when demand_qty > 0 then demand_qty else 0 end) as positive_qty,
        sort_array(
            filter(
                collect_list(case when demand_qty > 0 then demand_qty end),
                value -> value is not null
            )
        ) as positive_values
    from history
    group by cutoff_date, test_start, location_id, category_name
),

group_calendar_stats as (
    select
        cutoff_date,
        location_id,
        category_name,
        month(demand_month) as month_number,
        count(*) as calendar_rows,
        sum(case when demand_qty > 0 then 1 else 0 end) as calendar_positive_rows
    from history
    group by cutoff_date, location_id, category_name, month(demand_month)
),

target_months as (
    select
        c.cutoff_date,
        c.test_start,
        add_months(c.test_start, offsets.month_offset) as target_month
    from cutoffs as c
    cross join values (0), (1), (2) as offsets(month_offset)
),

series_features as (
    select
        ls.cutoff_date,
        targets.target_month,
        ls.product_variant_id,
        ls.location_id,
        ls.category_name,
        ls.location_type,
        actual.demand_qty as actual_qty,
        ls.ma3_frozen,
        coalesce(prior_year.demand_qty, ls.ma3_frozen) as seasonal_series,
        (
            ls.recent_positive_rows
            + 12.0 * (
                (
                    ps.recent_positive_rows
                    / nullif(cast(ps.recent_rows as double), 0.0)
                )
                + (
                    gs.recent_positive_rows
                    / nullif(cast(gs.recent_rows as double), 0.0)
                )
            ) / 2.0
        ) / (ls.recent_rows + 12.0) as local_probability,
        case
            when ls.positive_rows >= 6 then ls.positive_values
            when ps.positive_rows > 0 then ps.positive_values
            else gs.positive_values
        end as positive_values,
        (
            ls.positive_qty
            + 6.0 * coalesce(
                ps.positive_qty / nullif(cast(ps.positive_rows as double), 0.0),
                gs.positive_qty / nullif(cast(gs.positive_rows as double), 0.0),
                0.0
            )
        ) / (ls.positive_rows + 6.0) as positive_mean,
        case
            when calendar.calendar_rows > 0 and gs.all_positive_rows > 0
                then least(
                    2.0,
                    greatest(
                        0.5,
                        (
                            calendar.calendar_positive_rows
                            / cast(calendar.calendar_rows as double)
                        ) / (
                            gs.all_positive_rows
                            / cast(gs.all_rows as double)
                        )
                    )
                )
            else 1.0
        end as season_factor
    from local_stats as ls
    inner join target_months as targets
        on ls.cutoff_date = targets.cutoff_date
    inner join product_stats as ps
        on ls.cutoff_date = ps.cutoff_date
        and ls.product_variant_id = ps.product_variant_id
    inner join group_stats as gs
        on ls.cutoff_date = gs.cutoff_date
        and ls.location_id = gs.location_id
        and ls.category_name = gs.category_name
    left join group_calendar_stats as calendar
        on ls.cutoff_date = calendar.cutoff_date
        and ls.location_id = calendar.location_id
        and ls.category_name = calendar.category_name
        and calendar.month_number = month(targets.target_month)
    inner join source_data as actual
        on targets.target_month = actual.demand_month
        and ls.product_variant_id = actual.product_variant_id
        and ls.location_id = actual.location_id
    left join source_data as prior_year
        on add_months(targets.target_month, -12) = prior_year.demand_month
        and ls.product_variant_id = prior_year.product_variant_id
        and ls.location_id = prior_year.location_id
),

candidate_features as (
    select
        *,
        least(1.0, local_probability * coalesce(season_factor, 1.0))
            as calendar_probability
    from series_features
),

candidate_predictions as (
    select
        *,
        case
            when local_probability <= 0.5 or size(positive_values) = 0 then 0.0
            else element_at(
                positive_values,
                greatest(
                    1,
                    cast(
                        ceil(
                            (1.0 - 0.5 / local_probability)
                            * size(positive_values)
                        ) as int
                    )
                )
            )
        end as hurdle_median,
        case
            when calendar_probability <= 0.5 or size(positive_values) = 0 then 0.0
            else element_at(
                positive_values,
                greatest(
                    1,
                    cast(
                        ceil(
                            (1.0 - 0.5 / calendar_probability)
                            * size(positive_values)
                        ) as int
                    )
                )
            )
        end as hurdle_calendar_median,
        calendar_probability * positive_mean as hurdle_calendar_mean
    from candidate_features
),

forecast_rows as (
    select
        cutoff_date,
        target_month,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        actual_qty,
        'zero' as method,
        0.0 as predicted_qty
    from candidate_predictions

    union all

    select
        cutoff_date,
        target_month,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        actual_qty,
        'ma3_frozen' as method,
        ma3_frozen as predicted_qty
    from candidate_predictions

    union all

    select
        cutoff_date,
        target_month,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        actual_qty,
        'seasonal_series' as method,
        seasonal_series as predicted_qty
    from candidate_predictions

    union all

    select
        cutoff_date,
        target_month,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        actual_qty,
        'hurdle_median' as method,
        hurdle_median as predicted_qty
    from candidate_predictions

    union all

    select
        cutoff_date,
        target_month,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        actual_qty,
        'hurdle_calendar_median' as method,
        hurdle_calendar_median as predicted_qty
    from candidate_predictions

    union all

    select
        cutoff_date,
        target_month,
        product_variant_id,
        location_id,
        category_name,
        location_type,
        actual_qty,
        'hurdle_calendar_mean' as method,
        hurdle_calendar_mean as predicted_qty
    from candidate_predictions
),

scoped_forecasts as (
    select
        'all' as scope,
        forecast_rows.*
    from forecast_rows

    union all

    select
        'store_all' as scope,
        forecast_rows.*
    from forecast_rows
    where location_type = 'store'
),

overall_metrics as (
    select
        cutoff_date,
        scope,
        cast(null as date) as target_month,
        'all_3m' as period,
        method,
        count(*) as evaluated_rows,
        count(distinct product_variant_id, location_id) as evaluated_series,
        sum(actual_qty) as actual_qty,
        sum(predicted_qty) as predicted_qty,
        100.0 * (sum(predicted_qty) - sum(actual_qty))
            / nullif(sum(actual_qty), 0.0) as bias_pct,
        100.0 * avg(case when predicted_qty > 0 then 1.0 else 0.0 end)
            as positive_forecast_pct,
        sum(case when actual_qty = 0 and predicted_qty > 0 then 1 else 0 end)
            as false_positive_rows,
        sum(case when actual_qty > 0 and predicted_qty = 0 then 1 else 0 end)
            as missed_sale_rows,
        100.0 * sum(abs(actual_qty - predicted_qty))
            / nullif(sum(actual_qty), 0.0) as sku_location_wape,
        avg(abs(actual_qty - predicted_qty)) as sku_location_mae
    from scoped_forecasts
    group by cutoff_date, scope, method
),

monthly_metrics as (
    select
        cutoff_date,
        scope,
        target_month,
        cast(target_month as string) as period,
        method,
        count(*) as evaluated_rows,
        count(distinct product_variant_id, location_id) as evaluated_series,
        sum(actual_qty) as actual_qty,
        sum(predicted_qty) as predicted_qty,
        100.0 * (sum(predicted_qty) - sum(actual_qty))
            / nullif(sum(actual_qty), 0.0) as bias_pct,
        100.0 * avg(case when predicted_qty > 0 then 1.0 else 0.0 end)
            as positive_forecast_pct,
        sum(case when actual_qty = 0 and predicted_qty > 0 then 1 else 0 end)
            as false_positive_rows,
        sum(case when actual_qty > 0 and predicted_qty = 0 then 1 else 0 end)
            as missed_sale_rows,
        100.0 * sum(abs(actual_qty - predicted_qty))
            / nullif(sum(actual_qty), 0.0) as sku_location_wape,
        avg(abs(actual_qty - predicted_qty)) as sku_location_mae
    from scoped_forecasts
    group by cutoff_date, scope, target_month, method
),

metrics as (
    select * from overall_metrics
    union all
    select * from monthly_metrics
),

metrics_with_reference as (
    select
        *,
        min(
            case
                when method in ('zero', 'ma3_frozen', 'seasonal_series')
                    then sku_location_wape
            end
        ) over (partition by cutoff_date, scope, period) as best_baseline_wape
    from metrics
),

period_checks as (
    select
        *,
        case
            when method like 'hurdle_%'
                and sku_location_wape < best_baseline_wape
                and positive_forecast_pct > 0 then 1
            else 0
        end as passes_this_period
    from metrics_with_reference
),

full_holdout as (
    select
        c.cutoff_date,
        count(distinct f.product_variant_id, f.location_id) as all_grid_series,
        sum(f.demand_qty) as all_grid_qty
    from cutoffs as c
    inner join source_data as f
        on c.test_start <= f.demand_month
        and c.cutoff_date >= f.demand_month
    group by c.cutoff_date
),

eligible_holdout as (
    select
        cutoff_date,
        count(distinct product_variant_id, location_id) as eligible_series,
        sum(actual_qty) as eligible_qty
    from candidate_predictions
    group by cutoff_date
),

coverage as (
    select
        full_holdout.cutoff_date,
        full_holdout.all_grid_series,
        eligible_holdout.eligible_series,
        100.0 * eligible_holdout.eligible_series / full_holdout.all_grid_series
            as eligible_series_pct,
        100.0 * eligible_holdout.eligible_qty
            / nullif(full_holdout.all_grid_qty, 0.0) as eligible_actual_qty_pct
    from full_holdout
    inner join eligible_holdout
        on full_holdout.cutoff_date = eligible_holdout.cutoff_date
),

final as (
    select
        checks.cutoff_date,
        checks.scope,
        checks.period,
        checks.method,
        coverage.all_grid_series,
        coverage.eligible_series,
        round(coverage.eligible_series_pct, 2) as eligible_series_pct,
        round(coverage.eligible_actual_qty_pct, 2) as eligible_actual_qty_pct,
        checks.evaluated_series,
        checks.evaluated_rows,
        round(checks.actual_qty, 2) as actual_qty,
        round(checks.predicted_qty, 2) as predicted_qty,
        round(checks.bias_pct, 2) as bias_pct,
        round(checks.positive_forecast_pct, 2) as positive_forecast_pct,
        checks.false_positive_rows,
        checks.missed_sale_rows,
        round(checks.sku_location_wape, 2) as sku_location_wape,
        round(checks.sku_location_mae, 3) as sku_location_mae,
        round(checks.best_baseline_wape, 2) as best_baseline_wape,
        checks.passes_this_period,
        case
            when checks.method like 'hurdle_%'
                and min(
                    case
                        when checks.scope = 'store_all'
                            then checks.passes_this_period
                        else 1
                    end
                ) over (partition by checks.method) = 1 then true
            else false
        end as passes_all_store_periods
    from period_checks as checks
    inner join coverage
        on checks.cutoff_date = coverage.cutoff_date
)

select *
from final
order by cutoff_date, scope, period, method
