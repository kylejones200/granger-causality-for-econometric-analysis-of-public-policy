"""Granger prep — differencing and rolling stats via DuckDB."""

import duckdb
import polars as pl


def apply_differencing(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """First difference per column using LAG() — replaces pandas .diff()."""
    work = df.with_row_index("_idx")
    diff_exprs = ", ".join(
        f'"{c}" - LAG("{c}", 1) OVER (ORDER BY _idx) AS "{c}_diff"' for c in columns
    )
    base_cols = ", ".join(f'"{c}"' for c in columns)
    return duckdb.sql(f"""
        SELECT {base_cols}, {diff_exprs}
        FROM work
        ORDER BY _idx
    """).pl()


def rolling_correlation(
    df: pl.DataFrame, col_x: str, col_y: str, window: int = 24
) -> pl.DataFrame:
    w = window - 1
    return duckdb.sql(f"""
        SELECT
            *,
            CORR("{col_x}", "{col_y}") OVER (
                ORDER BY _idx ROWS BETWEEN {w} PRECEDING AND CURRENT ROW
            ) AS rolling_corr
        FROM (SELECT *, ROW_NUMBER() OVER () AS _idx FROM df)
        ORDER BY _idx
    """).pl()
