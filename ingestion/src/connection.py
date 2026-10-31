"""JDBC connection to the source SQL Server.

Provides helpers to retrieve JDBC settings from a Databricks Secret Scope."""


def get_jdbc_config(dbutils, secret_scope: str = "retail-analytics-source") -> tuple:
    """Builds the JDBC URL and connection properties from credentials
    stored in the Secret Scope.

    Returns:
        A tuple `(jdbc_url, jdbc_props)` for use with `spark.read.jdbc`.
    """

    # Read each credential from the Secret Scope.
    # dbutils.secrets.get() retrieves the value by key name.
    # Values never appear in logs or output; Databricks masks them automatically.

    host = dbutils.secrets.get(scope=secret_scope, key="db-host")

    port = dbutils.secrets.get(scope=secret_scope, key="db-port")

    database = dbutils.secrets.get(scope=secret_scope, key="db-name")

    user = dbutils.secrets.get(scope=secret_scope, key="db-user")

    password = dbutils.secrets.get(scope=secret_scope, key="db-password")

    # Build the JDBC URL.
    # Standard Microsoft JDBC driver format for SQL Server:
    # jdbc:sqlserver://<host>:<port>;databaseName=<database_name>
    jdbc_url = (
        f"jdbc:sqlserver://{host}:{port}"
        f";databaseName={database}"
        f";encrypt=true"
        f";trustServerCertificate=true"
    )
    # encrypt=true enables an encrypted connection (recommended practice).
    # trustServerCertificate=true accepts a self-signed server certificate
    # (common in test/lab environments).

    # Build the properties dictionary, passed to `spark.read.jdbc()`.
    jdbc_props = {
        "user": user,
        "password": password,
        "driver": "com.microsoft.sqlserver.jdbc.SQLServerDriver",
        # The driver above is preinstalled on Databricks clusters.
        # No additional JAR upload is required.
    }

    return jdbc_url, jdbc_props


def test_connection(spark, dbutils, secret_scope: str = "retail-analytics-source") -> bool:
    """Tests SQL Server connectivity by reading `dbo.attributes`.

    Returns:
        `True` if the JDBC read succeeds, otherwise `False`.
    """
    try:
        jdbc_url, jdbc_props = get_jdbc_config(dbutils, secret_scope)

        # Read a small table from the database to test the connection.
        df = spark.read.jdbc(
            url=jdbc_url, table="dbo.attributes", properties=jdbc_props
        )

        row_count = df.count()
        print(f"[OK] Connection successful! dbo.attributes returned {row_count} rows.")
        print("First rows:")
        df.show(5)

        return True

    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("Check the following:")
        print("  1. Does the 'retail-analytics-source' Secret Scope exist and contain all 5 keys?")
        print("  2. Can the Databricks cluster reach the host on the specified port?")
        print("  3. Are the username and password correct?")
        return False
