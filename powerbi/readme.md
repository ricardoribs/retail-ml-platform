# Power BI analytics report

This module documents a Power BI consumption layer. Its semantic model queries Databricks Gold tables and reports on orders, payments, customers, regions, returns, inventory, and demand forecasting.

## Contents

1. [Open the report](#1-open-the-report)
2. [Connect to Databricks](#2-connect-to-databricks)
3. [Refresh the data](#3-refresh-the-data)
4. [Data architecture](#4-data-architecture)
5. [Semantic model](#5-semantic-model)
6. [Report pages](#6-report-pages)
7. [DAX measures](#7-dax-measures)
8. [Notes](#8-notes)

## 1. Open the report

### Prerequisites

- Power BI Desktop version 2026.08 or later.
- Access to the Databricks workspace and an active SQL Warehouse.
- A Databricks Personal Access Token.
- Read access to retail_analytics.default_marts.

Request the current report file from its maintainer. The Retail Analytics.pbix file contains the report, semantic model, measures, and layout. All tables use Import mode, so the report opens with the last refreshed data without requiring the SQL Warehouse for navigation.

## 2. Connect to Databricks

The connection uses the native Databricks connector through Databricks.Catalogs and authenticates with a Personal Access Token.

### Power Query parameters

| Parameter | Value | Description |
|---|---|---|
| DatabricksHost | <workspace-host> | SQL Warehouse server hostname, without https://. |
| DatabricksHTTPPath | /sql/1.0/warehouses/<warehouse-id> | SQL Warehouse HTTP path. |
| Catalog | retail_analytics | Unity Catalog catalog. |
| Schema | default_marts | Gold schema produced by dbt. |

Find the server hostname and HTTP path under SQL Warehouses > your warehouse > Connection details. Generate a token under Settings > Developer > Access tokens.

### Steps

1. Open Home > Transform data > Power Query Editor.
2. Select Manage Parameters and enter the values above.
3. Select Close & Apply.
4. When prompted, choose Personal Access Token and enter the token.
5. Set the privacy level to Organizational.

To change the token, open File > Options and settings > Data source settings > Edit Permissions.

## 3. Refresh the data

Data is stored in the semantic model and changes only when a refresh runs. The SQL Warehouse must be active. Refresh after the pipeline completes.

~~~mermaid
flowchart LR
    A[Daily Databricks pipeline<br/>8:00 AM] --> B[Gold tables updated]
    B --> C[Refresh in Power BI Desktop]
    C --> D[Save the report]
~~~

1. Confirm that retail_analytics_data_pipeline completed successfully in Databricks Workflows or in retail_analytics.bronze.ingestion_control.
2. Confirm that the SQL Warehouse is running.
3. Open the PBIX file in Power BI Desktop.
4. Select Home > Refresh to reload all tables.
5. Save the file.

To refresh one table, right-click it in the Data pane and select Refresh data.

## 4. Data architecture

Power BI is the analytics consumption layer of the data lakehouse and reads only the Gold layer.

~~~mermaid
flowchart LR
    A[SQL Server<br/>retail_analytics] -->|Spark JDBC| B[Bronze<br/>24 Delta tables]
    B -->|dbt| C[Silver<br/>staging and intermediate]
    C -->|dbt| D[Gold<br/>retail_analytics.default_marts]
    D -->|Import| E[Power BI semantic model]
    E --> F[Report<br/>7 pages]
~~~

See the [ingestion guide](../ingestion/readme.md) for Bronze and the [dbt guide](../dbt/readme.md) for Silver and Gold transformations.

### Table sources

| Power BI table | Gold model | Grain |
|---|---|---|
| Calendar | dim_date | One row per date. |
| Customers | dim_customer | One row per customer. |
| Employees | dim_employee | One row per employee; record -1 represents e-commerce. |
| Locations | dim_location | One row per operating location. |
| Products | dim_product | One row per product variant. |
| Orders | fct_orders | One row per order. |
| Payments | fct_payments | One row per payment. |
| Sold Items | fct_sales | One row per order item. |
| Returns | fct_return_transactions | One row per return. |
| Returned Items | fct_returns | One row per returned item. |
| Inventory and Replenishment | fct_inventory_replenishment_snapshot | One row per snapshot, variant, and location. |
| Suppliers | dim_supplier | One row per supplier. |

Each Power Query M query uses Table.RenameColumns to translate technical Gold columns into business-friendly labels. Update the matching query whenever a Gold column is renamed.

## 5. Semantic model

The model uses a star schema. Dimensions filter fact tables through one-to-many relationships with single-direction cross-filtering.

~~~mermaid
flowchart TB
    CAL[Calendar] --> ORD[Orders]
    CAL --> PAY[Payments]
    CAL --> SOLD[Sold Items]
    CAL --> RET[Returns]
    CAL --> RITEM[Returned Items]
    CAL --> INV[Inventory and Replenishment]
    CUST[Customers] --> ORD
    CUST --> PAY
    CUST --> SOLD
    EMP[Employees] --> ORD
    EMP --> SOLD
    LOC[Locations] --> ORD
    LOC --> INV
    PROD[Products] --> SOLD
    PROD --> RITEM
    PROD --> INV
    SUP[Suppliers] --> INV
~~~

### Relationships

Date, customer, employee, location, product, and supplier dimensions filter the corresponding fact tables. Relationships from Sold Items and Payments to Orders are inactive and are enabled on demand by measures in the Order Details display folder with USERELATIONSHIP.

## 6. Report pages

The report has seven pages at 1280 × 720 resolution.

| # | Page | Purpose |
|---:|---|---|
| 1 | Executive Summary | Current-year management KPIs compared with the previous year. |
| 2 | Orders and Payments | Commercial and financial order performance. |
| 3 | Order Details | Installments, categories, cancellations, and seasonality. |
| 4 | Regions and Channels | Performance by geography and sales channel. |
| 5 | Customers | Customer value, repeat purchases, and retention. |
| 6 | Returns | Return volume, value, and root causes. |
| 7 | Inventory | Inventory position and replenishment needs. |

Pages 4 and 5 use both the customer sales region and the order's operating location. These are different concepts and must not be summed or compared directly.

## 7. DAX measures

The model contains 112 measures:

| Display folder | Measures | Contents |
|---|---:|---|
| Time Comparisons | 29 | YTD, prior periods, and year-over-year comparisons. |
| Customers | 21 | Recency, frequency, repeat purchases, and omnichannel profiles. |
| Orders and Sales | 18 | Orders, gross revenue, discounts, items, and cancellations. |
| Payments | 17 | Net revenue, average order value, status, and installments. |
| Inventory and Replenishment | 14 | Snapshots, alerts, coverage, and purchase suggestions. |
| Returns | 6 | Return quantity, value, and rate. |
| Order Details | 5 | Items and payments per order with USERELATIONSHIP. |

### Main business rules

| Measure | Rule |
|---|---|
| Recognized Net Revenue | Amount paid minus completed returns, floored at zero per order. |
| Paid Orders | Distinct count of orders with status paid. |
| Official Average Order Value | Recognized net revenue divided by paid orders. |
| Purchasing Customers | Distinct customers with at least one paid order. |
| Sold Item Value | Sum of paid order-item lines; it is not the same as net revenue. |
| Completed Return Value | Sum of refunds with completed status. |
| Current Snapshot Date | Latest snapshot on or before the selected date. |

When adding a measure, place it in the Measures table, assign a display folder and format, and reuse the model's reference date.

## 8. Notes

- The SQL Warehouse must be running during refresh. Outside refreshes, the report opens with its imported data.
- Never commit the Databricks token. It is stored in Power BI Desktop data source settings.