"""Web application server for Marina on Databricks Apps."""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from flask import Flask, jsonify, render_template, request

if __package__:
    from .src import llm, prompts, retriever, safety
else:  # Databricks Apps runs `python app.py` from the application root.
    from src import llm, prompts, retriever, safety

MAX_QUESTION_LENGTH = 2400
HIGH_CONFIDENCE_MIN_OBSERVATIONS = 3
MEDIUM_CONFIDENCE_MIN_OBSERVATIONS = 2

app = Flask(__name__, template_folder="templates")


def _json_value(value: Any) -> Any:
    """Convert connector values to types supported by jsonify."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class DatabricksMartProvider:
    """Query forecast outputs used by Marina's visualizations."""

    def __init__(self) -> None:
        self.catalog = os.environ.get("MARINA_CATALOG", "retail_analytics")
        self.schema = os.environ.get("MARINA_MARTS_SCHEMA", "default_marts")
        configured_product_id = os.environ.get("MARINA_SEASONALITY_PRODUCT_ID", "")
        self.product_id = (
            int(configured_product_id) if configured_product_id.strip() else None
        )

    @property
    def configured(self) -> bool:
        return all(
            (
                os.environ.get("DATABRICKS_SERVER_HOSTNAME")
                or os.environ.get("DATABRICKS_HOST"),
                os.environ.get("DATABRICKS_HTTP_PATH")
                or os.environ.get("DATABRICKS_WAREHOUSE_ID"),
                os.environ.get("DATABRICKS_CLIENT_ID"),
                os.environ.get("DATABRICKS_CLIENT_SECRET"),
            )
        )

    def _table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    def _server_hostname(self) -> str:
        hostname = (
            os.environ.get("DATABRICKS_SERVER_HOSTNAME")
            or os.environ["DATABRICKS_HOST"]
        )
        return hostname.removeprefix("https://").removeprefix("http://")

    def _http_path(self) -> str:
        configured_path = os.environ.get("DATABRICKS_HTTP_PATH")
        if configured_path:
            return configured_path
        return f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}"

    def _query(self, query: str) -> list[dict[str, Any]]:
        if not self.configured:
            raise RuntimeError("Databricks connection is not configured for Marina.")
        try:
            from databricks import sql
            from databricks.sdk.core import Config, oauth_service_principal
        except ImportError as exc:  # pragma: no cover - exercised in deployment
            raise RuntimeError(
                "The databricks-sql-connector and databricks-sdk dependencies "
                "are not installed."
            ) from exc

        server_hostname = self._server_hostname()

        def credential_provider():
            config = Config(
                host=f"https://{server_hostname}",
                client_id=os.environ["DATABRICKS_CLIENT_ID"],
                client_secret=os.environ["DATABRICKS_CLIENT_SECRET"],
            )
            return oauth_service_principal(config)

        connection = sql.connect(
            server_hostname=server_hostname,
            http_path=self._http_path(),
            credentials_provider=credential_provider,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                return [
                    {column: _json_value(value) for column, value in zip(columns, row)}
                    for row in cursor.fetchall()
                ]
        finally:
            connection.close()

    def forecast(self) -> dict[str, Any]:
        """Loads only the latest fully approved forecast."""
        forecast = self._query(
            f"""
            WITH latest_run AS (
                SELECT run_id, MAX(created_at) AS run_created_at
                FROM {self._table("agg_location_demand_forecast")}
                GROUP BY run_id
                HAVING COUNT(DISTINCT horizon_month) = 3
                ORDER BY run_created_at DESC, run_id DESC
                LIMIT 1
            )
            SELECT f.run_id,
                   date_format(f.forecast_month, 'yyyy-MM') AS period,
                   SUM(CAST(f.forecast_demand_qty AS DOUBLE)) AS demand_qty,
                   COUNT(DISTINCT f.location_id) AS location_count,
                   COUNT(DISTINCT CASE
                       WHEN f.location_type = 'store' THEN f.location_id
                   END) AS store_count,
                   COUNT(DISTINCT CASE
                       WHEN f.location_type = 'warehouse' THEN f.location_id
                   END) AS distribution_center_count
            FROM {self._table("agg_location_demand_forecast")} f
            INNER JOIN latest_run r ON f.run_id = r.run_id
            GROUP BY f.run_id, f.forecast_month
            ORDER BY f.forecast_month
            """
        )
        scope = forecast[0] if forecast else {}
        return {
            "horizons": len(forecast),
            "scope": "location_capacity_planning",
            "location_count": scope.get("location_count", 0),
            "store_count": scope.get("store_count", 0),
            "distribution_center_count": scope.get("distribution_center_count", 0),
            "series": forecast,
        }

    def forecast_publication(self) -> dict[str, Any]:
        """Summarizes whether the latest snapshot can power operational rankings."""
        rows = self._query(
            f"""
            WITH latest_run AS (
                SELECT run_id, MAX(created_at) AS run_created_at
                FROM {self._table("fct_forecast_series_publication")}
                GROUP BY run_id
                ORDER BY run_created_at DESC, run_id DESC
                LIMIT 1
            )
            SELECT p.run_id,
                   COUNT(*) AS series_count,
                   SUM(CASE
                           WHEN p.publication_status = 'approved_forecast' THEN 1
                           ELSE 0
                       END)
                       AS approved_series_count,
                   SUM(CASE WHEN p.review_required THEN 1 ELSE 0 END)
                       AS review_required_series_count,
                   'product_location_decision' AS product_location_scope,
                   'location_capacity_planning' AS planning_forecast_scope,
                   CASE
                     WHEN EXISTS (
                         SELECT 1
                         FROM {self._table("agg_location_demand_forecast")} aggregate
                         WHERE aggregate.run_id = p.run_id
                         GROUP BY aggregate.run_id
                         HAVING COUNT(DISTINCT aggregate.horizon_month) = 3
                     ) THEN true
                     ELSE false
                   END AS actionable_for_rankings
            FROM {self._table("fct_forecast_series_publication")} p
            INNER JOIN latest_run r ON p.run_id = r.run_id
            GROUP BY p.run_id
            """
        )
        return rows[0] if rows else {}

    def seasonality(self) -> dict[str, Any]:
        """Loads the featured seasonal product or the configured product."""
        if self.product_id is None:
            selected_product_query = f"""
                SELECT product_variant_id
                FROM {self._table("agg_product_seasonality")}
                GROUP BY product_variant_id
                HAVING MIN(observations) >= {HIGH_CONFIDENCE_MIN_OBSERVATIONS}
                ORDER BY
                    STDDEV_POP(avg_monthly_demand_qty) /
                        NULLIF(AVG(overall_avg_monthly_demand_qty), 0) DESC NULLS LAST,
                    product_variant_id
                LIMIT 1
            """
            selection_method = "highest_cv_minimum_monthly_observations"
            selection_description = (
                "Automatically featured: greatest month-to-month variation among products with "
                "at least three observations in each month."
            )
        else:
            selected_product_query = f"SELECT {self.product_id} AS product_variant_id"
            selection_method = "configured_product"
            selection_description = (
                "Featured product configured in Marina."
            )

        seasonality = self._query(
            f"""
            WITH selected_product AS (
                {selected_product_query}
            )
            SELECT seasonality.product_variant_id, seasonality.sku,
                   seasonality.product_name, seasonality.month_number,
                   seasonality.seasonality_index, seasonality.seasonality_class,
                   seasonality.observations,
                   STDDEV_POP(avg_monthly_demand_qty) OVER () /
                     NULLIF(AVG(overall_avg_monthly_demand_qty) OVER (), 0) AS cv
            FROM {self._table("agg_product_seasonality")} seasonality
            INNER JOIN selected_product
                ON seasonality.product_variant_id = selected_product.product_variant_id
            ORDER BY seasonality.month_number
            """
        )
        cv = next(
            (row.get("cv") for row in seasonality if row.get("cv") is not None),
            None,
        )
        selected_product = next(iter(seasonality), {})
        observations = [
            int(row["observations"])
            for row in seasonality
            if row.get("observations") is not None
        ]
        min_observations = min(observations, default=None)
        confidence = self._seasonality_confidence(min_observations)
        return {
            "product_id": selected_product.get("product_variant_id"),
            "sku": selected_product.get("sku"),
            "product_name": selected_product.get("product_name"),
            "cv": cv,
            "min_observations_per_month": min_observations,
            "confidence": confidence,
            "selection_method": selection_method,
            "selection_description": selection_description,
            "series": seasonality,
        }

    @staticmethod
    def _seasonality_confidence(min_observations: int | None) -> str | None:
        """Classifies confidence using the lowest number of monthly observations."""
        if min_observations is None:
            return None
        if min_observations >= HIGH_CONFIDENCE_MIN_OBSERVATIONS:
            return "high"
        if min_observations >= MEDIUM_CONFIDENCE_MIN_OBSERVATIONS:
            return "medium"
        return "low"

    def _location_growth(self, comparison_type: str) -> list[dict[str, Any]]:
        """Loads a growth ranking by location type."""
        return self._query(
            f"""
            WITH actionable_run AS (
                SELECT run_id, MAX(created_at) AS run_created_at
                FROM {self._table("agg_location_demand_forecast")}
                GROUP BY run_id
                HAVING COUNT(DISTINCT horizon_month) = 3
                ORDER BY run_created_at DESC, run_id DESC
                LIMIT 1
            )
            SELECT location_name, city, state,
                   actual_start_month, actual_end_date AS actual_end_month,
                   forecast_start_month, forecast_end_month,
                   growth_abs_qty, growth_pct, growth_rank_abs, growth_rank_pct
            FROM {self._table("agg_location_forecast_growth")} growth
            INNER JOIN actionable_run run
                ON growth.source_forecast_run_id = run.run_id
            WHERE comparison_type = '{comparison_type}'
              AND (growth_rank_abs <= 5 OR growth_rank_pct <= 5)
            ORDER BY growth_rank_abs, growth_abs_qty DESC
            """
        )

    def locations(self) -> list[dict[str, Any]]:
        """Loads the store growth ranking."""
        return self._location_growth("store")

    def distribution_centers(self) -> list[dict[str, Any]]:
        """Loads the growth ranking for existing distribution centers."""
        return self._location_growth("distribution_center")

    def recommendations(self) -> list[dict[str, Any]]:
        """Loads the distribution center recommendation ranking."""
        recommendations = self._query(
            f"""
            WITH actionable_run AS (
                SELECT run_id, MAX(created_at) AS run_created_at
                FROM {self._table("agg_location_demand_forecast")}
                GROUP BY run_id
                HAVING COUNT(DISTINCT horizon_month) = 3
                ORDER BY run_created_at DESC, run_id DESC
                LIMIT 1
            )
            SELECT macro_region, has_dedicated_dc,
                   currently_covered_by_name,
                   actual_last_3m_demand_qty,
                   forecast_next_3m_demand_qty,
                   growth_abs_qty, growth_pct, growth_base_status,
                   source_forecast_run_id,
                   candidate_rank, recommendation_score,
                   technical_recommendation, recommendation_reason
            FROM {self._table("agg_distribution_center_recommendation")} recommendation
            INNER JOIN actionable_run run
                ON recommendation.source_forecast_run_id = run.run_id
            ORDER BY candidate_rank ASC NULLS LAST, macro_region
            """
        )
        return recommendations

    def visuals(self) -> dict[str, Any]:
        """Loads the time series used by the technical dashboard."""
        forecast = self.forecast()
        seasonality = self.seasonality()
        locations = self.locations()
        distribution_centers = self.distribution_centers()
        recommendations = self.recommendations()
        top_candidate = next(
            (row for row in recommendations if row.get("candidate_rank") is not None),
            None,
        )
        return {
            "status": "ready",
            "forecast": forecast,
            "forecast_policy": self.forecast_publication(),
            "seasonality": seasonality,
            "locations": locations,
            "distribution_centers": distribution_centers,
            "recommendations": recommendations,
            "top_candidate": top_candidate,
        }

    def model_metrics(self) -> list[dict[str, Any]]:
        """Loads metrics for all models evaluated in the latest run."""
        return self._query(
            f"""
            WITH latest_run AS (
                SELECT run_id, MAX(created_at) AS run_created_at
                FROM {self._table("fct_forecast_model_metrics")}
                GROUP BY run_id
                ORDER BY run_created_at DESC, run_id DESC
                LIMIT 1
            )
            SELECT m.run_id, m.model_name, m.model_family,
                   m.evaluation_window,
                   CAST(m.mae AS DOUBLE) AS mae,
                   CAST(m.rmse AS DOUBLE) AS rmse,
                   CAST(m.mape AS DOUBLE) AS mape,
                   CAST(m.wape AS DOUBLE) AS wape,
                   m.is_best_model, m.created_at
            FROM {self._table("fct_forecast_model_metrics")} m
            INNER JOIN latest_run r ON m.run_id = r.run_id
            ORDER BY m.is_best_model DESC,
                     m.wape ASC NULLS LAST,
                     m.rmse ASC NULLS LAST,
                     m.model_name
            """
        )

    def seasonality_ranking(self) -> list[dict[str, Any]]:
        """Loads the products with the greatest seasonal variation."""
        return self._query(
            f"""
            SELECT product_variant_id,
                   MAX(sku) AS sku,
                   MAX(product_name) AS product_name,
                   STDDEV_POP(avg_monthly_demand_qty) /
                     NULLIF(AVG(overall_avg_monthly_demand_qty), 0) AS cv,
                   MAX(seasonality_index) AS peak_index,
                   MIN(observations) AS min_observations_per_month,
                   CASE
                     WHEN MIN(observations) >= 3 THEN 'high'
                     WHEN MIN(observations) >= 2 THEN 'medium'
                     ELSE 'low'
                   END AS confidence,
                   COUNT(*) AS observed_months
            FROM {self._table("agg_product_seasonality")}
            GROUP BY product_variant_id
            ORDER BY cv DESC NULLS LAST, product_variant_id
            LIMIT 10
            """
        )


class LocalDemoMartProvider:
    """Mock data for reviewing Marina visuals locally."""

    catalog = "local_demo"
    schema = "marts"

    @staticmethod
    def forecast() -> dict[str, Any]:
        return {
            "horizons": 3,
            "scope": "location_capacity_planning",
            "location_count": 6,
            "store_count": 3,
            "distribution_center_count": 3,
            "series": [
                {
                    "run_id": "local_demo_2026_08",
                    "period": "2026-09",
                    "demand_qty": 5790,
                },
                {
                    "run_id": "local_demo_2026_08",
                    "period": "2026-10",
                    "demand_qty": 8934,
                },
                {
                    "run_id": "local_demo_2026_08",
                    "period": "2026-11",
                    "demand_qty": 12061,
                },
            ],
        }

    @staticmethod
    def forecast_publication() -> dict[str, Any]:
        return {
            "run_id": "local_demo_2026_08",
            "series_count": 6054,
            "approved_series_count": 6054,
            "review_required_series_count": 1845,
            "product_location_scope": "product_location_decision",
            "planning_forecast_scope": "location_capacity_planning",
            "actionable_for_rankings": True,
        }

    @staticmethod
    def seasonality() -> dict[str, Any]:
        return {
            "product_id": 608,
            "sku": "SKU-462189",
            "product_name": "Outboard Motor 2640",
            "cv": 0.908,
            "min_observations_per_month": 3,
            "confidence": "high",
            "selection_method": "highest_cv_minimum_monthly_observations",
            "selection_description": (
                "Automatically featured: greatest month-to-month variation among products with "
                "at least three observations in each month."
            ),
            "series": [
                {"month_number": 1, "seasonality_index": 1.12},
                {"month_number": 2, "seasonality_index": 1.03},
                {"month_number": 3, "seasonality_index": 1.09},
                {"month_number": 4, "seasonality_index": 0.98},
                {"month_number": 5, "seasonality_index": 0.91},
                {"month_number": 6, "seasonality_index": 0.93},
                {"month_number": 7, "seasonality_index": 0.88},
                {"month_number": 8, "seasonality_index": 0.91},
                {"month_number": 9, "seasonality_index": 0.89},
                {"month_number": 10, "seasonality_index": 0.94},
                {"month_number": 11, "seasonality_index": 1.01},
                {"month_number": 12, "seasonality_index": 1.22},
            ],
        }

    @staticmethod
    def locations() -> list[dict[str, Any]]:
        return [
            {
                "location_name": "Store 1",
                "city": "Goiânia",
                "state": "GO",
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
                "growth_abs_qty": 2044,
                "growth_pct": 21.4,
                "growth_rank_abs": 1,
                "growth_rank_pct": 1,
            },
            {
                "location_name": "Store 2",
                "city": "Campo Grande",
                "state": "MS",
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
                "growth_abs_qty": 1764,
                "growth_pct": 18.8,
                "growth_rank_abs": 2,
                "growth_rank_pct": 2,
            },
            {
                "location_name": "Store 3",
                "city": "Brasília",
                "state": "DF",
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
                "growth_abs_qty": 1712,
                "growth_pct": 17.2,
                "growth_rank_abs": 3,
                "growth_rank_pct": 3,
            },
        ]

    @staticmethod
    def distribution_centers() -> list[dict[str, Any]]:
        return [
            {
                "location_name": "Distribution Center 1",
                "city": "Goiânia",
                "state": "GO",
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
                "growth_abs_qty": 5112,
                "growth_pct": 19.1,
                "growth_rank_abs": 1,
                "growth_rank_pct": 1,
            },
            {
                "location_name": "Distribution Center 2",
                "city": "Brasília",
                "state": "DF",
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
                "growth_abs_qty": 4485,
                "growth_pct": 16.4,
                "growth_rank_abs": 2,
                "growth_rank_pct": 2,
            },
            {
                "location_name": "Distribution Center 3",
                "city": "Campo Grande",
                "state": "MS",
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
                "growth_abs_qty": 3490,
                "growth_pct": 12.7,
                "growth_rank_abs": 3,
                "growth_rank_pct": 3,
            },
        ]

    @staticmethod
    def recommendations() -> list[dict[str, Any]]:
        return [
            {
                "macro_region": "Central-West",
                "has_dedicated_dc": False,
                "currently_covered_by_name": "Distribution Center 1",
                "actual_last_3m_demand_qty": 3680,
                "forecast_next_3m_demand_qty": 4005,
                "growth_abs_qty": 325,
                "growth_pct": 8.8,
                "growth_base_status": "positive_base",
                "source_forecast_run_id": "local_demo_2026_08",
                "candidate_rank": 1,
                "recommendation_score": 0.76,
                "technical_recommendation": "evaluate",
                "recommendation_reason": "Forecast growth with current coverage.",
            },
            {
                "macro_region": "Southeast",
                "has_dedicated_dc": False,
                "currently_covered_by_name": "Distribution Center 2",
                "actual_last_3m_demand_qty": 3525,
                "forecast_next_3m_demand_qty": 3860,
                "growth_abs_qty": 335,
                "growth_pct": 9.5,
                "growth_base_status": "positive_base",
                "source_forecast_run_id": "local_demo_2026_08",
                "candidate_rank": 2,
                "recommendation_score": 0.7,
                "technical_recommendation": "evaluate",
                "recommendation_reason": "Forecast growth with current coverage.",
            },
        ]

    @staticmethod
    def model_metrics() -> list[dict[str, Any]]:
        return [
            {
                "run_id": "local_demo_2026_08",
                "model_name": "seasonal_naive",
                "model_family": "baseline",
                "evaluation_window": "local_demo",
                "mae": 41.2,
                "rmse": 72.8,
                "mape": 0.19,
                "wape": 0.13,
                "is_best_model": True,
            }
        ]

    @staticmethod
    def seasonality_ranking() -> list[dict[str, Any]]:
        return [
            {
                "product_variant_id": 608,
                "sku": "SKU-462189",
                "product_name": "Outboard Motor 2640",
                "cv": 0.908,
                "peak_index": 1.22,
                "min_observations_per_month": 3,
                "confidence": "high",
                "observed_months": 12,
            }
        ]

    def visuals(self) -> dict[str, Any]:
        recommendations = self.recommendations()
        return {
            "status": "ready",
            "data_mode": "local_demo",
            "forecast": self.forecast(),
            "forecast_policy": self.forecast_publication(),
            "seasonality": self.seasonality(),
            "locations": self.locations(),
            "distribution_centers": self.distribution_centers(),
            "recommendations": recommendations,
            "top_candidate": recommendations[0],
        }


def _local_demo_data_enabled() -> bool:
    """Indicates whether local runs should use demonstration data."""
    return os.environ.get("MARINA_USE_MOCK_DATA", "").casefold() in {
        "1",
        "true",
        "yes",
    }


mart_provider: Any = (
    LocalDemoMartProvider() if _local_demo_data_enabled() else DatabricksMartProvider()
)

MART_CONTEXT_TABLES = {
    "forecast": "agg_location_demand_forecast",
    "forecast_policy": "fct_forecast_series_publication",
    "model_metrics": "fct_forecast_model_metrics",
    "seasonality": "agg_product_seasonality",
    "seasonality_ranking": "agg_product_seasonality",
    "locations": "agg_location_forecast_growth",
    "distribution_centers": "agg_location_forecast_growth",
    "recommendations": "agg_distribution_center_recommendation",
}

ALL_OPERATIONAL_CONTEXT_KEYS = tuple(MART_CONTEXT_TABLES)

OPERATIONAL_CONTEXT_LOADERS = {
    "forecast": "forecast",
    "forecast_policy": "forecast_publication",
    "model_metrics": "model_metrics",
    "seasonality": "seasonality",
    "seasonality_ranking": "seasonality_ranking",
    "locations": "locations",
    "distribution_centers": "distribution_centers",
    "recommendations": "recommendations",
}


def _context_value_has_data(key: str, value: Any) -> bool:
    """Distinguishes structural payloads from series that contain rows."""
    if key in {"forecast", "seasonality"}:
        return isinstance(value, dict) and bool(value.get("series"))
    return bool(value)


def _has_operational_context(operational_context: dict[str, Any]) -> bool:
    """Indicates whether at least one operational source returned usable data."""
    return any(
        _context_value_has_data(key, operational_context.get(key))
        for key in MART_CONTEXT_TABLES
    )


def _normalized_question(question: str) -> str:
    """Normalizes accents and case for deterministic context selection."""
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", question.lower())
        if not unicodedata.combining(character)
    )


def _operational_context_keys(question: str) -> tuple[str, ...]:
    """Selects only marts related to the question topic."""
    normalized = f" {_normalized_question(question)} "
    if any(
        term in normalized
        for term in ("current metrics", "current data", "current operational data", "indicadores atuais", "dados operacionais", "resultados atuais")
    ):
        return ALL_OPERATIONAL_CONTEXT_KEYS

    keys: list[str] = []
    forecast_terms = (
        "forecast",
        "demand forecast",
        "forecast demand",
        "forecast horizon",
        "next three months",
        "september",
        "october",
        "november",
        "cumulative",
        "revenue",
        "units forecast",
        "previsao",
        "demanda prevista",
        "horizonte",
        "setembro",
        "outubro",
        "novembro",
        "acumulad",
        "faturamento",
        "unidades previstas",
    )
    scope_terms = (
        "network",
        "consolidated",
        "scope",
        "location",
        "how many stores",
        "how many distribution centers",
        "stores and distribution centers",
        "entire network",
        "rede",
        "consolid",
        "escopo",
        "localidade",
        "quantas lojas",
        "quantos cds",
        "lojas e cds",
        "lojas e cd",
        "toda",
    )
    asks_forecast_scope = any(term in normalized for term in scope_terms)
    asks_location_demand = ("loja" in normalized or "store" in normalized or "stores" in normalized) and any(
        term in normalized
        for term in ("highest demand", "forecast demand per store", "each store", "maior demanda", "demanda prevista por loja", "cada loja")
    )
    if any(term in normalized for term in forecast_terms) and (
        not asks_location_demand or asks_forecast_scope
    ):
        keys.extend(("forecast", "forecast_policy"))
    elif asks_forecast_scope:
        keys.extend(("forecast", "forecast_policy"))

    if any(
        term in normalized
        for term in (
            "sku", "series", "review", "order", "place an order",
            "serie",
            "revisao",
            "revisoes",
            "revisar",
            "pedido",
            "automatic",
            "automatica",
            "emitir",
            "aprovad",
            "planning review",
        )
    ):
        keys.append("forecast_policy")

    if any(
        term in normalized
        for term in (
            "model", "models", "modelo",
            "mae",
            "rmse",
            "mape",
            "wape",
            "metrica",
            "precis",
            "acur",
            "erro",
        )
    ):
        keys.append("model_metrics")

    if any(
        term in normalized
        for term in (
            "seasonality", "seasonal", "coefficient of variation", "variation", "peak", "low", "featured product", "sazon",
            "coeficiente de variacao",
            " cv ",
            "pico",
            "vale",
            "oscilacao",
            "cores do grafico",
            "produto em destaque",
        )
    ):
        ranking_question = any(
            term in normalized
            for term in ("products", "ranking", "most seasonal", "greatest seasonal", "produtos", "mais sazonal", "maior sazonal")
        )
        keys.append("seasonality_ranking" if ranking_question else "seasonality")

    location_question = ("loja" in normalized or "store" in normalized or "stores" in normalized) and any(
        term in normalized
        for term in (
            "growth", "grew", "increase", "leader", "highest", "crescimento",
            "cresceu",
            "crescer",
            "ranking",
            "aumento",
            "lider",
            "lidera",
        )
    )
    if location_question or "crescimento absoluto" in normalized:
        keys.extend(("locations", "forecast_policy"))

    has_distribution_center = bool(
        re.search(r"\b(?:cds?|distribution centers?)\b", normalized) or "centro de distribuicao" in normalized
    )
    distribution_center_question = has_distribution_center and any(
        term in normalized
        for term in (
            "growth", "grew", "increase", "leader", "highest", "crescimento",
            "cresceu",
            "crescer",
            "ranking",
            "aumento",
            "lider",
            "lidera",
        )
    )
    is_new_distribution_center_question = any(
        term in normalized
        for term in ("new distribution center", "new dc", "screening", "region", "macro-region", "candidate", "novo cd", "triagem", "regiao", "macrorregiao", "candidata")
    )
    if distribution_center_question and not is_new_distribution_center_question:
        keys.extend(("distribution_centers", "forecast_policy"))

    if any(
        term in normalized
        for term in (
            "distribution center", "open a distribution center", "distribution center screening", "centro de distribuicao",
            "new distribution center", "new dc", "screening", "region", "macro-region", "candidate", "novo cd",
            "abrir um cd",
            "macrorregiao",
            "candidata",
            "triagem",
            "regiao",
        )
    ):
        keys.extend(("recommendations", "forecast_policy"))

    sku_location_question = any(
        term in normalized
        for term in (
            "sku", "series", "review", "order", "place an order",
            "product-location", "product-store", "sku-store", "product location", "replenishment", "reorder", "produto-localidade",
            "produto localidade",
            "produto-loja",
            "produto loja",
            "reposicao",
        )
    )
    if sku_location_question and any(
        term in normalized for term in ("forecast", "automatic", "replenishment", "reorder", "previs", "reposicao")
    ):
        keys.extend(("forecast", "forecast_policy"))
    return tuple(dict.fromkeys(keys))


def _load_operational_context(
    provider: Any, requested_keys: tuple[str, ...]
) -> dict[str, Any]:
    """Loads each operational group without losing others when one fails."""
    context: dict[str, Any] = {}
    for key in requested_keys:
        method_name = OPERATIONAL_CONTEXT_LOADERS[key]
        loader = getattr(provider, method_name, None)
        if loader is None:
            continue
        try:
            context[key] = loader()
        except Exception as exc:
            app.logger.warning("Failed to load Marina %s: %s", method_name, exc)
    return context


def _prepare_operational_context(
    question: str, operational_context: dict[str, Any]
) -> dict[str, Any]:
    """Removes conflicting metrics and clarifies definitions before prompting."""
    prepared = dict(operational_context)
    normalized = _normalized_question(question)

    forecast = operational_context.get("forecast")
    if isinstance(forecast, dict) and forecast.get("series"):
        location_count = forecast.get("location_count", 0)
        store_count = forecast.get("store_count", 0)
        distribution_center_count = forecast.get("distribution_center_count", 0)
        prepared["forecast"] = {
            **forecast,
            "definition": (
                "demand_qty is the forecast monthly total for "
                f"{location_count} network locations ({store_count} stores and "
                f"{distribution_center_count} distribution centers); each month is independent, "
                "values are not cumulative and do not represent revenue"
            ),
        }

    for context_key in ("locations", "distribution_centers"):
        ranked_locations = operational_context.get(context_key)
        if not ranked_locations:
            continue
        percentage = "percent" in normalized
        rank_field = "growth_rank_pct" if percentage else "growth_rank_abs"
        value_field = "growth_pct" if percentage else "growth_abs_qty"
        ranked = [row for row in ranked_locations if row.get(rank_field) is not None]
        if ranked:
            leader = min(
                ranked,
                key=lambda row: (row[rank_field], row.get("location_name", "")),
            )
            selected_leader = {
                "location_name": leader.get("location_name"),
                "city": leader.get("city"),
                "state": leader.get("state"),
                value_field: leader.get(value_field),
                rank_field: leader.get(rank_field),
                "ranking_basis": "percentage" if percentage else "absolute",
            }
            period_fields = (
                "actual_start_month",
                "actual_end_month",
                "forecast_start_month",
                "forecast_end_month",
            )
            for field in period_fields:
                if leader.get(field) is not None:
                    selected_leader[field] = leader[field]
            if any(field in selected_leader for field in period_fields):
                selected_leader["comparison_definition"] = (
                    "growth_abs_qty is total forecast demand for the future period minus "
                    "realized demand from the preceding period"
                )
            prepared[context_key] = [selected_leader]

    seasonality = operational_context.get("seasonality")
    if isinstance(seasonality, dict) and seasonality.get("confidence") == "high":
        prepared["seasonality"] = {
            **seasonality,
            "confidence_definition": (
                "at least three observations in each calendar month; it does not represent "
                "accuracy, robustness, or statistical reliability"
            ),
        }

    recommendations = operational_context.get("recommendations")
    if recommendations:
        prepared["recommendations"] = [
            {
                **recommendation,
                "dashboard_value_definition": (
                    "the card displays forecast_next_3m_demand_qty: "
                    "total forecast demand for the next three months, in "
                    "units; it is not a score, growth, or an automated "
                    "opening decision"
                ),
            }
            for recommendation in recommendations
        ]
    return prepared


def _source_with_suffix(sources: list[str], suffix: str) -> str | None:
    """Finds an available source without inventing identifiers in the response."""
    normalized_suffix = suffix.replace("\\", "/").casefold()
    return next(
        (
            source
            for source in sources
            if source.replace("\\", "/").casefold().endswith(normalized_suffix)
        ),
        None,
    )


def _format_number(value: Any, decimals: int = 3) -> str:
    """Formats a numeric metric using the English interface convention."""
    formatted = f"{float(value):,.{decimals}f}"
    return formatted


def _format_units(value: Any) -> str:
    """Format units without decimals when the value is an integer."""
    numeric_value = float(value)
    decimals = 0 if numeric_value.is_integer() else 2
    return _format_number(numeric_value, decimals)


def _format_month_range(start: Any, end: Any) -> str | None:
    """Convert a date range to the compact month form used by Marina."""
    try:
        start_date = (
            start if isinstance(start, date) else date.fromisoformat(str(start)[:10])
        )
        end_date = end if isinstance(end, date) else date.fromisoformat(str(end)[:10])
    except (TypeError, ValueError):
        return None

    month_names = (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    )
    start_label = month_names[start_date.month - 1]
    end_label = month_names[end_date.month - 1]
    if start_date.year == end_date.year:
        return f"{start_label}-{end_label}/{start_date.year % 100:02d}"
    return (
        f"{start_label}/{start_date.year % 100:02d}-"
        f"{end_label}/{end_date.year % 100:02d}"
    )


def _model_label(model_name: Any) -> str:
    """Converts persisted identifiers into short labels for the response."""
    labels = {
        "baseline_zero": "Baseline zero",
        "baseline_seasonal": "Seasonal baseline",
        "baseline_ma3": "MA3",
        "baseline_ma6": "MA6",
        "gradient_boosted_trees": "Gradient Boosted Trees",
        "random_forest": "Random Forest",
        "ridge_regression": "Ridge",
    }
    normalized = str(model_name or "")
    return labels.get(normalized, normalized.replace("_", " ").title())


def _best_metric_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Selects the persisted winner, falling back to the lowest finite WAPE."""
    persisted_winner = next((row for row in rows if row.get("is_best_model")), None)
    if persisted_winner is not None:
        return persisted_winner
    eligible = [row for row in rows if row.get("wape") is not None]
    return min(eligible, key=lambda row: float(row["wape"]), default=None)


def _deterministic_answer(
    question: str,
    operational_context: dict[str, Any],
    sources: list[str],
) -> tuple[str, list[str]] | None:
    """Answers critical facts directly from marts and documented contracts.

    These answers do not depend on the LLM reproducing numbers, selecting the correct source,
    or interpreting a sensitive operational policy.
    """
    normalized = _normalized_question(question)
    growth_source = _source_with_suffix(sources, ".agg_location_forecast_growth")
    growth_terms = ("growth", "grew", "increase", "leader", "highest", "growth", "grew", "increase", "leader", "highest", "crescimento", "cresceu", "crescer", "aumento", "lider")

    def growth_leader_answer(
        context_key: str, location_label: str
    ) -> tuple[str, list[str]] | None:
        leaders = operational_context.get(context_key) or []
        if not leaders or growth_source is None:
            return None
        leader = leaders[0]
        growth = leader.get("growth_abs_qty")
        actual_period = _format_month_range(
            leader.get("actual_start_month"), leader.get("actual_end_month")
        )
        forecast_period = _format_month_range(
            leader.get("forecast_start_month"), leader.get("forecast_end_month")
        )
        if growth is None or actual_period is None or forecast_period is None:
            return None
        return (
            f"{leader.get('location_name')} was the {location_label} with the largest "
            f"forecast increase: {_format_units(growth)} units. The comparison is "
            f"the forecast for {forecast_period} versus realized demand for "
            f"{actual_period}.",
            [growth_source],
        )

    asks_store_growth = ("loja" in normalized or "store" in normalized or "stores" in normalized) and any(
        term in normalized for term in growth_terms
    )
    if asks_store_growth:
        answer = growth_leader_answer("locations", "store")
        if answer is not None:
            return answer

    has_distribution_center = bool(
        re.search(r"\b(?:cds?|distribution centers?)\b", normalized) or "centro de distribuicao" in normalized
    )
    asks_new_distribution_center = any(
        term in normalized for term in ("new dc", "screening", "region", "macro-region", "new distribution center", "new dc", "screening", "region", "macro-region", "candidate", "novo cd", "triagem", "regiao", "macrorregiao")
    )
    asks_distribution_center_growth = has_distribution_center and any(
        term in normalized for term in growth_terms
    )
    if asks_distribution_center_growth and not asks_new_distribution_center:
        answer = growth_leader_answer("distribution_centers", "distribution center")
        if answer is not None:
            return answer

    recommendation_source = _source_with_suffix(
        sources, ".agg_distribution_center_recommendation"
    )
    recommendations = operational_context.get("recommendations") or []
    asks_automatic_new_dc = asks_new_distribution_center and any(
        term in normalized for term in ("automatic", "open", "recommend", "abrir", "recomenda")
    )
    if asks_automatic_new_dc and recommendation_source:
        return (
            "No. Screening only prioritizes further analysis; opening a distribution center "
            "depends on capacity, service levels, and costs, with a human decision.",
            [recommendation_source],
        )

    if asks_new_distribution_center and recommendations and recommendation_source:
        ranked_recommendations = [
            recommendation
            for recommendation in recommendations
            if recommendation.get("candidate_rank") is not None
        ]
        if ranked_recommendations:
            leader = min(
                ranked_recommendations,
                key=lambda recommendation: int(recommendation["candidate_rank"]),
            )
            forecast_qty = leader.get("forecast_next_3m_demand_qty")
            if forecast_qty is not None:
                return (
                    f"{leader.get('macro_region')} leads the screening. The displayed value "
                    f"is {_format_units(forecast_qty)} units forecast for the "
                    "next three months; this is not a growth rate or score, and it "
                    "is not an automated opening decision.",
                    [recommendation_source],
                )

    metric_rows = operational_context.get("model_metrics") or []
    metric_source = _source_with_suffix(sources, ".fct_forecast_model_metrics")
    asks_model_decision = ("modelo" in normalized or "model" in normalized) and any(
        term in normalized
        for term in ("best", "lowest", "wape", "mape", "accuracy", "accurate", "reliable", "melhor", "menor", "precis", "confiav")
    )
    if asks_model_decision and metric_rows and metric_source:
        best_overall = _best_metric_row(metric_rows)
        if best_overall is None:
            return None

        if "regress" in normalized:
            regression_rows = [
                row
                for row in metric_rows
                if not str(row.get("model_name", "")).startswith("baseline_")
                and row.get("wape") is not None
            ]
            best_regression = min(
                regression_rows,
                key=lambda row: float(row["wape"]),
                default=None,
            )
            if best_regression is None:
                return None
            baselines = {
                str(row.get("model_name")): row
                for row in metric_rows
                if row.get("wape") is not None
            }
            comparisons = []
            for model_name, label in (
                ("baseline_zero", "zero"),
                ("baseline_seasonal", "seasonal"),
            ):
                baseline = baselines.get(model_name)
                if baseline is not None:
                    comparisons.append(
                        f"{label} ({_format_number(baseline['wape'])}%)"
                    )
            comparison_text = (
                f" It did not beat {' or '.join(comparisons)}, so it was not "
                "promoted to the final forecast."
                if comparisons
                else " Error remains high and should be interpreted with caution."
            )
            return (
                f"{_model_label(best_regression.get('model_name'))} was the regression "
                 "with the lowest WAPE in the latest run, "
                f"{_format_number(best_regression['wape'])}%."
                f"{comparison_text}",
                [metric_source],
            )

        if any(term in normalized for term in ("accuracy", "accurate", "reliable", "precis", "confiav")):
            wape = best_overall.get("wape")
            metric_text = (
                f", with WAPE of {_format_number(wape)}%" if wape is not None else ""
            )
            return (
                 "No. "
                f"{_model_label(best_overall.get('model_name'))} performed best only "
                f"among the models evaluated in the latest run{metric_text}; the error is "
                "high and does not demonstrate operational accuracy or reliability.",
                [metric_source],
            )

        requested_metrics = []
        for field, label, decimals, percent in (
            ("wape", "WAPE", 3, True),
            ("mape", "MAPE", 3, True),
            ("mae", "MAE", 3, False),
            ("rmse", "RMSE", 3, False),
        ):
            if field in normalized and best_overall.get(field) is not None:
                suffix = "%" if percent else ""
                formatted_value = _format_number(best_overall[field], decimals)
                requested_metrics.append(f"{label} {formatted_value}{suffix}")
        metrics_text = ", ".join(requested_metrics)
        if metrics_text:
            metrics_text = f": {metrics_text}"
        return (
            f"{_model_label(best_overall.get('model_name'))} had the lowest error among "
            f"the models evaluated in the latest run{metrics_text}. WAPE is high, "
            "so interpret the result with caution; it does not prove high accuracy.",
            [metric_source],
        )

    policy = operational_context.get("forecast_policy") or {}
    policy_source = _source_with_suffix(sources, ".fct_forecast_series_publication")
    asks_review_count = ("series" in normalized or "serie" in normalized) and ("review" in normalized or "revisa" in normalized)
    if asks_review_count and policy_source:
        review_count = policy.get("review_required_series_count")
        if review_count is not None:
            return (
                f"In the latest run, {_format_units(review_count)} SKU-store series "
                "require review.",
                [policy_source],
            )

    asks_automatic_sku_use = any(
        term in normalized
        for term in ("sku", "series", "review", "order", "place an order", "product-location", "product store", "product-store", "sku-store", "product-location", "product-store", "sku-store", "product location", "replenishment", "reorder", "produto-localidade", "produto loja", "produto-loja")
    ) and any(term in normalized for term in ("automatic", "replenishment", "reorder", "reposicao"))
    if asks_automatic_sku_use and policy and policy_source:
        review_count = int(policy.get("review_required_series_count") or 0)
        if review_count > 0:
            used_sources = [policy_source]
            aggregate_source = _source_with_suffix(
                sources, ".agg_location_demand_forecast"
            )
            if policy.get("actionable_for_rankings") and aggregate_source:
                used_sources.append(aggregate_source)
            return (
                 "No. SKU-store forecasts require review and must not trigger "
                 "automatic replenishment; the location-level aggregate remains "
                "valid for capacity planning and rankings.",
                used_sources,
            )

    readme_source = _source_with_suffix(sources, "ml/demand_forecasting/readme.md")
    seasonality_source = _source_with_suffix(sources, ".agg_product_seasonality")
    asks_cv_precision = any(
        term in normalized for term in ("cv", "variation", "coefficient of variation", "oscilacao", "coeficiente de variacao")
    ) and any(term in normalized for term in ("prove", "accuracy", "accurate", "reliable", "prova", "precis", "confiav"))
    if asks_cv_precision and seasonality_source and metric_source:
        return (
             "No. The variation index (CV) measures historical demand variation; it "
             "does not measure forecast accuracy. Accuracy is "
            "evaluated in backtests using metrics such as WAPE and MAPE.",
            [seasonality_source, metric_source],
        )

    if ("season" in normalized or "sazon" in normalized) and ("cause" in normalized or "caus" in normalized):
        used_sources = [
            source for source in (seasonality_source, readme_source) if source
        ]
        if used_sources:
            return (
                 "The data identify a monthly pattern but do not determine "
                 "its cause. There is no documented evidence linking it to weather, "
                "seasons, or commercial events.",
                used_sources,
            )

    if any(term in normalized for term in ("feature", "features", "attribute", "atributo")) and readme_source:
        return (
             "The forecast uses lag_1_qty, lag_2_qty, and lag_3_qty; "
             "rolling_3m_avg_qty, rolling_6m_avg_qty, and rolling_3m_sum_qty; plus "
             "year_number, quarter_number, and month_number. Lags and rolling windows use "
            "only months before the forecast month; no future actuals are used.",
            [readme_source],
        )

    asks_temporal_leakage = (
        ("backtest" in normalized or "histor" in normalized or "history" in normalized)
    ) and any(term in normalized for term in ("price", "inventory", "lead time", "preco", "estoque"))
    if asks_temporal_leakage and readme_source:
        return (
             "Using current price, inventory, or lead time in a backtest would cause temporal "
             "leakage. Price and frequency must be lagged; current inventory "
             "does not represent historical monthly balances, and current lead time cannot "
            "be projected backward.",
            [readme_source],
        )

    app_readme_source = _source_with_suffix(sources, "apps/marina/readme.md")
    asks_credential_protection = ("credential" in normalized or "credencia" in normalized) and any(
        term in normalized for term in ("protect", "protection", "prevent", "avoid", "proteg", "evitar", "previn")
    )
    if asks_credential_protection and app_readme_source:
        return (
             "Never commit tokens or credentials. In the application runtime, use "
             "service principal OAuth and valueFrom resources, apply least "
            "privilege, and do not expose values in code, responses, or logs.",
            [app_readme_source],
        )

    return None


def _mart_sources(operational_context: dict, provider: Any) -> list[str]:
    """Returns mart table names included in the operational context."""
    catalog = getattr(
        provider, "catalog", os.environ.get("MARINA_CATALOG", "retail_analytics")
    )
    schema = getattr(
        provider,
        "schema",
        os.environ.get("MARINA_MARTS_SCHEMA", "default_marts"),
    )
    sources = []
    for key, table in MART_CONTEXT_TABLES.items():
        if not _context_value_has_data(key, operational_context.get(key)):
            continue
        source = f"{catalog}.{schema}.{table}"
        if source not in sources:
            sources.append(source)
    return sources


def _combine_sources(
    primary_sources: list[str], secondary_sources: list[str]
) -> list[str]:
    """Combines primary and secondary sources without duplicates."""
    combined = list(primary_sources)
    for source in secondary_sources:
        if source not in combined:
            combined.append(source)
    return combined


@app.get("/")
def index():
    """Renderiza a interface HTML da Marina."""
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    """Serves the logo through the legacy path used by browsers."""
    return app.send_static_file("logo-retail-analytics.svg")


@app.get("/health")
def health():
    """Simple health-check endpoint for the Databricks App."""
    return jsonify(status="ok", app="marina")


@app.get("/api/visuals")
def visuals():
    """Returns series for the technical visuals on the home page."""
    try:
        return jsonify(mart_provider.visuals())
    except Exception as exc:  # keep the UI available when marts are unavailable
        app.logger.warning("Could not load metrics: %s", exc)
        return jsonify(status="unavailable", error=str(exc)), 200


@app.post("/api/chat")
def chat():
    """Answers a question using retrieval over the knowledge base."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = {}

    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify(error="Enter a question."), 400
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify(error="The question exceeds the 2,400-character limit."), 400
    if safety.is_disallowed_question(question):
        return jsonify(
            status="refused",
            answer=safety.SAFE_REFUSAL_MESSAGE,
            sources=[],
            question=question,
        )

    try:
        chunks = retriever.retrieve(question)
    except Exception as exc:
        app.logger.warning("Failed to query Marina index: %s", exc)
        return jsonify(
            status="unavailable",
            answer=(
                "I could not query the knowledge base right now. "
                "Please try again shortly."
            ),
            sources=[],
            question=question,
        ), 200

    requested_context_keys = _operational_context_keys(question)
    operational_context = _load_operational_context(
        mart_provider, requested_context_keys
    )
    operational_context = _prepare_operational_context(question, operational_context)

    doc_sources = retriever.sources_for(chunks)
    sources = _combine_sources(
        _mart_sources(operational_context, mart_provider), doc_sources
    )

    deterministic = _deterministic_answer(
        question,
        operational_context,
        sources,
    )
    if deterministic is not None:
        answer, used_sources = deterministic
        return jsonify(
            status="answered",
            answer=answer,
            sources=used_sources,
            question=question,
        )

    if not retriever.has_sufficient_context(chunks) and not _has_operational_context(
        operational_context
    ):
        return jsonify(
            status="no_context",
            answer=prompts.format_no_context_response(doc_sources),
            sources=[],
            question=question,
        )

    messages = prompts.build_messages(
        question,
        chunks,
        operational_context=operational_context,
        available_sources=sources,
    )
    try:
        generated_answer = llm.call_llm(messages)
    except Exception as exc:
        app.logger.warning("Failed to call Marina's LLM: %s", exc)
        return jsonify(
            status="unavailable",
            answer=(
                "I found relevant context, but could not generate the "
                "an answer right now. Please try again shortly."
            ),
            sources=[],
            question=question,
        ), 200

    if safety.exposes_protected_instructions(generated_answer, prompts.SYSTEM_PROMPT):
        app.logger.warning(
            "Blocked Marina response because it exposed internal instructions."
        )
        return jsonify(
            status="refused",
            answer=safety.SAFE_REFUSAL_MESSAGE,
            sources=[],
            question=question,
        )

    answer, used_sources = prompts.parse_answer_sources(generated_answer, sources)
    issues = prompts.validate_answer(question, answer, used_sources)
    if issues:
        app.logger.info(
            "Sent Marina response for correction; rules=%s; sources=%s",
            issues,
            used_sources,
        )
        try:
            generated_answer = llm.call_llm(
                prompts.build_correction_messages(messages, generated_answer, issues)
            )
            if safety.exposes_protected_instructions(
                generated_answer, prompts.SYSTEM_PROMPT
            ):
                app.logger.warning(
                    "Blocked Marina correction because it exposed internal instructions."
                )
                return jsonify(
                    status="refused",
                    answer=safety.SAFE_REFUSAL_MESSAGE,
                    sources=[],
                    question=question,
                )
            answer, used_sources = prompts.parse_answer_sources(
                generated_answer, sources
            )
            issues = prompts.validate_answer(question, answer, used_sources)
        except Exception as exc:
            app.logger.warning("Failed to correct Marina's answer: %s", exc)

    if issues:
        app.logger.warning(
            "Rejected Marina correction; rules=%s; sources=%s",
            issues,
            used_sources,
        )
        return jsonify(
            status="no_context",
            answer=prompts.UNVERIFIED_SOURCES_MESSAGE,
            sources=[],
            question=question,
        )
    return jsonify(
        status="answered",
        answer=answer,
        sources=used_sources,
        question=question,
    )


if __name__ == "__main__":
    port = int(
        os.environ.get(
            "DATABRICKS_APP_PORT",
            os.environ.get("PORT", "8000"),
        )
    )
    app.run(host="0.0.0.0", port=port)
