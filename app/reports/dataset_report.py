"""Build private, deterministic dataset profiles and polished PDF reports."""
from __future__ import annotations

import io
import logging
import math
import re
from collections import OrderedDict
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select, text

from app.config import get_settings
from app.storage.database import app_session, readonly_session
from app.storage.models import ColumnCatalog, Source

log = logging.getLogger(__name__)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_PII_HINTS = (
    "email", "phone", "mobile", "ssn", "social_security", "street", "address",
    "postcode", "postal", "zip_code", "birth", "dob", "first_name", "last_name",
    "full_name", "customer_name", "person_name", "username", "user_name", "password",
    "secret", "token", "credit_card", "card_number", "bank_account", "passport",
    "license_number", "contact", "latitude", "longitude",
)
_PROFILE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_PROFILE_CACHE_LIMIT = 24
_CHART_SAMPLE_LIMIT = 2500


def invalidate_dataset_profile(source_id: str) -> None:
    """Remove a deleted or changed source from the in-process profile cache."""
    _PROFILE_CACHE.pop(source_id, None)


def build_dataset_profile(source_id: str) -> dict[str, Any]:
    """Profile the selected source using bounded aggregates and chart samples."""
    cached = _PROFILE_CACHE.get(source_id)
    if cached:
        _PROFILE_CACHE.move_to_end(source_id)
        return cached

    with app_session() as session:
        source = session.get(Source, source_id)
        if not source or not source.is_active:
            raise KeyError("Dataset not found")
        source_meta = {
            "id": source.id,
            "name": source.name,
            "schema_name": source.schema_name or "quantai_data",
            "table_name": source.table_name or "",
            "row_count": int(source.row_count or 0),
            "col_count": int(source.col_count or 0),
            "file_path": source.file_path,
        }
        catalog = [
            {
                "name": column.safe_name,
                "label": column.original_name or column.safe_name,
                "type": column.col_type or "unknown",
                "mysql_type": column.mysql_type or "",
                "null_pct": float(column.null_pct or 0),
                "distinct_count": int(column.distinct_count or 0),
                "minimum": column.col_min,
                "maximum": column.col_max,
                "is_date": bool(column.is_date_col),
                "is_key": bool(column.is_key_col),
                "is_pii": bool(column.is_pii),
                "ordinal": int(column.ordinal_position or 0),
            }
            for column in session.execute(
                select(ColumnCatalog)
                .where(ColumnCatalog.source_id == source_id)
                .order_by(ColumnCatalog.ordinal_position)
            ).scalars().all()
        ]

    schema = _quote(source_meta["schema_name"])
    table = _quote(source_meta["table_name"])
    public_catalog = [column for column in catalog if not _is_sensitive(column)]
    has_multiple_currencies = any(
        "currency" in column["name"].lower() and column["distinct_count"] > 1
        for column in public_catalog
    )
    money_columns_excluded = [
        column for column in public_catalog
        if has_multiple_currencies and _is_money_metric(column["name"])
    ]
    numeric_columns = [
        column for column in public_catalog
        if _is_analytical_measure(column)
        and column not in money_columns_excluded
    ]
    numeric_columns.sort(key=lambda column: (_metric_priority(column), column["ordinal"]))
    selected_metrics = numeric_columns[:8]
    category_columns = [
        column for column in public_catalog
        if column["type"] in ("str", "bool") and 1 < column["distinct_count"] <= 100
    ]
    category_columns.sort(key=lambda column: (_category_priority(column), column["ordinal"]))
    selected_categories = category_columns[:4]
    date_columns = [
        column for column in public_catalog
        if column.get("is_date") or column["type"] in ("date", "datetime") or _is_temporal_column(column)
    ]

    measures: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    charts: list[dict[str, Any]] = []
    distribution_summary: dict[str, Any] | None = None

    with readonly_session() as connection:
        if selected_metrics:
            expressions: list[str] = []
            for index, column in enumerate(selected_metrics):
                name = _quote(column["name"])
                expressions.extend((
                    f"AVG({name}) AS `avg_{index}`",
                    f"MIN({name}) AS `min_{index}`",
                    f"MAX({name}) AS `max_{index}`",
                    f"STDDEV_POP({name}) AS `std_{index}`",
                    f"COUNT({name}) AS `non_null_{index}`",
                ))
            row = connection.execute(text(
                f"SELECT {', '.join(expressions)} FROM {schema}.{table}"
            )).mappings().one()
            for index, column in enumerate(selected_metrics):
                measures.append({
                    "name": column["name"],
                    "label": column["label"],
                    "average": _number(row.get(f"avg_{index}")),
                    "minimum": _number(row.get(f"min_{index}")),
                    "maximum": _number(row.get(f"max_{index}")),
                    "stddev": _number(row.get(f"std_{index}")),
                    "non_null": int(row.get(f"non_null_{index}") or 0),
                    "null_pct": column["null_pct"],
                })

        for column in selected_categories[:4]:
            name = _quote(column["name"])
            rows = connection.execute(text(
                f"SELECT {name} AS value, COUNT(*) AS frequency "
                f"FROM {schema}.{table} WHERE {name} IS NOT NULL "
                f"GROUP BY {name} ORDER BY frequency DESC LIMIT 8"
            )).mappings().all()
            values = [
                {"value": _label(row["value"]), "count": int(row["frequency"] or 0)}
                for row in rows
            ]
            if values:
                categories.append({"name": column["name"], "label": column["label"], "values": values})

        # Time series: aggregated by time bucket (year or month) using the full source.
        if date_columns and measures:
            date_col = date_columns[0]
            metric = selected_metrics[0]
            date_name = _quote(date_col["name"])
            metric_name = _quote(metric["name"])
            is_year_col = "year" in date_col["name"].lower() or date_col.get("type") in ("int", "float")
            bucket_expr = date_name if is_year_col else f"DATE_FORMAT({date_name}, '%Y-%m')"

            rows = connection.execute(text(
                f"SELECT {bucket_expr} AS bucket, AVG({metric_name}) AS value "
                f"FROM {schema}.{table} WHERE {date_name} IS NOT NULL AND {metric_name} IS NOT NULL "
                "GROUP BY bucket ORDER BY bucket DESC LIMIT 36"
            )).mappings().all()
            rows = list(reversed(rows))
            if len(rows) >= 2:
                time_unit = "yearly" if is_year_col else "monthly"
                charts.append({
                    "kind": "line",
                    "title": f"{metric['label']} over time",
                    "caption": f"Average {metric['label']} across the dataset, grouped by {date_col['label']} ({time_unit}).",
                    "labels": [_label(row["bucket"]) for row in rows],
                    "values": [_number(row["value"]) or 0 for row in rows],
                })

        # Compare meaningful numeric measures across a low-cardinality category.
        if selected_categories and measures:
            category = next(
                (column for column in selected_categories if _usable_group_dimension(column["name"])),
                None,
            )
            if category is None:
                category = selected_categories[0]
            metric = selected_metrics[0]
            category_name = _quote(category["name"])
            metric_name = _quote(metric["name"])

            # Check if dataset has a units column — filter to dominant unit to prevent mixing Dollars, %, etc.
            unit_col = next((c for c in public_catalog if c["name"].lower() in ("units", "unit", "currency")), None)
            unit_filter = ""
            unit_params: dict[str, Any] = {}
            unit_suffix = ""
            if unit_col:
                unit_col_name = _quote(unit_col["name"])
                top_u_row = connection.execute(text(
                    f"SELECT {unit_col_name} AS u, COUNT(*) AS cnt FROM {schema}.{table} "
                    f"WHERE {unit_col_name} IS NOT NULL GROUP BY {unit_col_name} ORDER BY cnt DESC LIMIT 1"
                )).mappings().first()
                if top_u_row and top_u_row["u"]:
                    unit_filter = f" AND {unit_col_name} = :unit_val"
                    unit_params["unit_val"] = top_u_row["u"]
                    unit_suffix = f" [{top_u_row['u']}]"

            # Check if dataset has hierarchical aggregation levels — filter to top level to avoid double-counting
            agg_col = next((c for c in public_catalog if any(k in c["name"].lower() for k in ("aggregation", "level_nzsioc", "agg_level"))), None)
            agg_filter = ""
            agg_params: dict[str, Any] = {}
            if agg_col:
                agg_col_name = _quote(agg_col["name"])
                top_a_row = connection.execute(text(
                    f"SELECT {agg_col_name} AS a, COUNT(*) AS cnt FROM {schema}.{table} "
                    f"WHERE {agg_col_name} IS NOT NULL GROUP BY {agg_col_name} ORDER BY cnt DESC LIMIT 1"
                )).mappings().first()
                if top_a_row and top_a_row["a"]:
                    agg_filter = f" AND {agg_col_name} = :agg_val"
                    agg_params["agg_val"] = top_a_row["a"]

            where_clause = f"{category_name} IS NOT NULL AND {metric_name} IS NOT NULL{unit_filter}{agg_filter}"
            sql_params = {**unit_params, **agg_params}

            rows = connection.execute(text(
                f"SELECT {category_name} AS label, AVG({metric_name}) AS value, COUNT(*) AS records "
                f"FROM {schema}.{table} WHERE {where_clause} "
                f"GROUP BY {category_name} ORDER BY value DESC LIMIT 10"
            ), sql_params).mappings().all()

            if len(rows) >= 2:
                caption_notes = [f"Each bar shows the average {metric['label']} for a {category['label']} group."]
                if unit_suffix:
                    caption_notes.append(f"Filtered to unit: {unit_params.get('unit_val')} to ensure valid comparison.")
                if agg_params:
                    caption_notes.append(f"Standardized on {agg_params.get('agg_val')} to prevent subcategory double-counting.")

                chart = {
                    "kind": "bar",
                    "title": f"Average {metric['label']}{unit_suffix} by {category['label']}",
                    "caption": " ".join(caption_notes),
                    "labels": [_label(row["label"]) for row in rows],
                    "values": [_number(row["value"]) or 0 for row in rows],
                    "overall_average": measures[0].get("average"),
                    "group_counts": [
                        {"label": _label(row["label"]), "count": int(row["records"] or 0)}
                        for row in rows
                    ],
                    "metric": f"{metric['label']}{unit_suffix}",
                    "dimension": category["label"],
                    "top_group": _label(rows[0]["label"]),
                    "top_value": _number(rows[0]["value"]),
                    "top_count": int(rows[0]["records"] or 0),
                }
                if not any(existing["title"] == chart["title"] for existing in charts):
                    charts.append(chart)

        # A representative sample for continuous numeric distributions (never temporal years or degenerate samples)
        distribution_metric = next(
            (item for item in selected_metrics if item["distinct_count"] > 10 and not _is_temporal_column(item)),
            None,
        )
        if distribution_metric:
            metric_name = _quote(distribution_metric["name"])
            samples = connection.execute(text(
                f"SELECT {metric_name} AS value FROM {schema}.{table} "
                f"WHERE {metric_name} IS NOT NULL LIMIT {_CHART_SAMPLE_LIMIT}"
            )).scalars().all()
            numeric_sample = np.asarray([float(value) for value in samples if _is_number(value)], dtype=float)
            if numeric_sample.size >= 10 and float(np.std(numeric_sample)) > 1e-6 and float(numeric_sample.max() - numeric_sample.min()) > 1e-6:
                q1, median, q3 = np.percentile(numeric_sample, [25, 50, 75]).tolist()
                distribution_summary = {
                    "metric": distribution_metric["label"],
                    "sample_count": int(numeric_sample.size),
                    "q1": _number(q1),
                    "median": _number(median),
                    "q3": _number(q3),
                    "iqr": _number(q3 - q1),
                }
                counts, edges = np.histogram(numeric_sample, bins=min(10, max(4, int(np.sqrt(numeric_sample.size)))))
                labels = [_format_bin_range(edges[i], edges[i + 1]) for i in range(len(counts))]
                charts.append({
                    "kind": "histogram",
                    "title": f"Distribution of {distribution_metric['label']}",
                    "caption": f"The histogram shows {distribution_metric['label']} values across a sample of {numeric_sample.size:,} rows.",
                    "labels": labels,
                    "values": [int(value) for value in counts],
                    "sample_count": int(numeric_sample.size),
                })

        # Add a frequency plot when a useful multi-value category is present.
        # This makes the categorical distribution visible alongside numeric
        # trends without producing a redundant chart for two-value flags.
        frequency_category = next(
            (item for item in categories if len(item.get("values", [])) >= 3),
            None,
        )
        if frequency_category and len(charts) < 3:
            charts.append({
                "kind": "bar",
                "title": f"Most common {frequency_category['label']} values",
                "caption": f"Record counts for the most common {frequency_category['label']} labels across the dataset.",
                "labels": [item["value"] for item in frequency_category["values"][:8]],
                "values": [item["count"] for item in frequency_category["values"][:8]],
                "y_label": "Record count",
            })

        # A category frequency chart gives categorical-only datasets a useful visual.
        if not charts and categories:
            category = categories[0]
            charts.append({
                "kind": "bar",
                "title": f"Most common {category['label']} values",
                "caption": f"The bars compare record counts for the most common {category['label']} values.",
                "labels": [item["value"] for item in category["values"]],
                "values": [item["count"] for item in category["values"]],
            })

        # Add correlation heatmap when >= 2 numeric measures exist
        if len(selected_metrics) >= 2 and len(charts) < 4:
            corr_metrics = selected_metrics[:6]
            metric_cols_str = ", ".join(_quote(m["name"]) for m in corr_metrics)
            try:
                corr_rows = connection.execute(text(
                    f"SELECT {metric_cols_str} FROM {schema}.{table} "
                    f"LIMIT {_CHART_SAMPLE_LIMIT}"
                )).mappings().all()
                if len(corr_rows) >= 5:
                    corr_df = pd.DataFrame([dict(r) for r in corr_rows]).apply(pd.to_numeric, errors="coerce")
                    corr_mat = corr_df.corr().fillna(0).round(2).values.tolist()
                    charts.append({
                        "kind": "heatmap",
                        "title": "Correlation Matrix",
                        "caption": "Pearson correlation coefficients across numeric variables.",
                        "labels": [m["label"] for m in corr_metrics],
                        "values": [1.0] * len(corr_metrics),
                        "matrix": corr_mat,
                    })
            except Exception:
                pass

        # Missing values breakdown when there are fields with unrecorded data
        missing_catalog = [c for c in catalog if c.get("null_pct", 0) > 0 and not _is_sensitive(c)]
        if missing_catalog and len(charts) < 4:
            missing_sorted = sorted(missing_catalog, key=lambda c: c["null_pct"], reverse=True)[:8]
            charts.append({
                "kind": "missing_bar",
                "title": "Missing Values Overview",
                "caption": "Percentage of unrecorded values across dataset fields.",
                "labels": [c["label"] for c in missing_sorted],
                "values": [c["null_pct"] for c in missing_sorted],
                "y_label": "Missing %",
            })

    row_count = source_meta["row_count"]
    col_count = source_meta["col_count"] or len(catalog)
    cell_count = row_count * col_count
    file_size = _file_size(source_meta["file_path"])
    if row_count >= 5000 or cell_count >= 100_000 or col_count >= 30 or file_size >= 5 * 1024 * 1024:
        scale = "large"
        content_sections = (
            3
            + math.ceil(max(len(catalog), 1) / 24)
            + math.ceil(max(len(measures), 1) / 5)
            + math.ceil(max(len(categories), 1) / 3)
            + len(charts)
            + int(len(charts) >= 2)
        )
        suggested_pages = min(13, max(5, content_sections))
    elif row_count >= 1000 or cell_count >= 20_000 or col_count >= 12 or file_size >= 1024 * 1024:
        scale = "medium"
        suggested_pages = 3 if len(charts) <= 1 else 4
    else:
        scale = "small"
        suggested_pages = 2 if not charts else 3

    high_missing = [column for column in catalog if column["null_pct"] >= 20 and not _is_sensitive(column)]
    any_missing = [column for column in catalog if column["null_pct"] > 0 and not _is_sensitive(column)]
    quality_notes = []
    metric_notes = []
    if money_columns_excluded:
        names = ", ".join(column["label"] for column in money_columns_excluded[:4])
        metric_notes.append(
            f"Combined averages for {names} were omitted because this dataset contains multiple currencies; normalize amounts before comparing them."
        )
    if high_missing:
        quality_notes.append(f"{len(high_missing)} field(s) have at least 20% missing values: " + ", ".join(c["label"] for c in high_missing[:8]) + ".")
    elif any_missing:
        quality_notes.append("Some fields contain missing values, but none exceed the 20% review threshold.")
    else:
        quality_notes.append("No missing values were reported in the imported dataset profile.")
    quality_notes.append(f"{sum(1 for c in catalog if _is_sensitive(c))} potentially sensitive field(s) were excluded from report examples and charts.")

    highlights = [f"The dataset contains {row_count:,} rows across {col_count:,} columns."]
    if measures:
        for measure in measures[:2]:
            if measure["average"] is not None:
                highlights.append(
                    f"{measure['label']} averages {_display(measure['average'])}, with observed values from "
                    f"{_display(measure['minimum'])} to {_display(measure['maximum'])}."
                )
    if categories and categories[0]["values"]:
        item = categories[0]["values"][0]
        highlights.append(f"The most common {categories[0]['label']} value is {item['value']} ({item['count']:,} rows).")
    if charts:
        for chart in charts:
            if chart.get("top_group") is not None:
                highlights.append(
                    f"{chart['top_group']} has the highest average {chart['metric']} "
                    f"({ _display(chart['top_value']) }) among the compared {chart['dimension']} groups."
                )
                break
    if high_missing:
        highlights.append(f"Review {len(high_missing)} field(s) with at least 20% missing values before drawing conclusions.")
    unique_highlights = list(dict.fromkeys(highlights))
    for chart in charts:
        chart["interpretation"] = _chart_interpretation(chart)
    summary = _build_executive_summary(
        row_count, col_count, measures, distribution_summary, charts, quality_notes, categories,
    )
    key_findings = []
    primary_measure = measures[0]["label"].casefold() if measures else None
    for finding in unique_highlights:
        normalized = finding.casefold()
        if finding.startswith("The dataset contains"):
            continue
        if primary_measure and primary_measure in normalized:
            continue
        if "has the highest average" in normalized or (high_missing and "review" in normalized):
            continue
        key_findings.append(finding)
    if not key_findings:
        key_findings = unique_highlights[:3]
    chart_summaries = [
        {
            "title": chart["title"],
            "caption": chart["caption"],
            "interpretation": chart.get("interpretation", ""),
            "kind": chart["kind"],
        }
        for chart in charts
    ]
    profile = {
        "source_id": source_meta["id"],
        "name": source_meta["name"],
        "row_count": row_count,
        "col_count": col_count,
        "file_size_bytes": file_size,
        "scale": scale,
        "suggested_pages": suggested_pages,
        "summary": summary,
        "highlights": key_findings[:6],
        "quality_notes": quality_notes,
        "metric_notes": metric_notes,
        "catalog": [
            {key: value for key, value in column.items() if key not in ("is_pii", "is_key")}
            for column in public_catalog[:80]
        ],
        "measures": measures,
        "distribution_summary": distribution_summary,
        "categories": categories,
        "charts": charts[:4],
        "chart_summaries": chart_summaries[:4],
        "generated_at": datetime.now().isoformat(timespec="minutes"),
    }
    _PROFILE_CACHE[source_id] = profile
    _PROFILE_CACHE.move_to_end(source_id)
    while len(_PROFILE_CACHE) > _PROFILE_CACHE_LIMIT:
        _PROFILE_CACHE.popitem(last=False)
    return profile


def _build_executive_summary(row_count, col_count, measures, distribution, charts, quality_notes, categories):
    """Summarize the dataset's scale, strongest measures, and evidence limits."""
    sentences = [f"This report profiles {row_count:,} records across {col_count:,} fields."]
    if measures:
        measure = measures[0]
        if measure.get("average") is not None:
            sentence = (
                f"{measure['label']} averages {_display(measure['average'])} across the full dataset, "
                f"with observed values from {_display(measure.get('minimum'))} to {_display(measure.get('maximum'))}."
            )
            if distribution:
                sentence += (
                    f" In a bounded sample of {distribution['sample_count']:,} records, the median is "
                    f"{_display(distribution.get('median'))}, and the middle half falls between "
                    f"{_display(distribution.get('q1'))} and {_display(distribution.get('q3'))}."
                )
            sentences.append(sentence)

    comparison = next((chart for chart in charts if chart.get("group_counts") and chart.get("values")), None)
    if comparison:
        groups = [
            (label, value, group)
            for label, value, group in zip(
                comparison.get("labels", []), comparison.get("values", []), comparison["group_counts"],
            )
            if value is not None
        ]
        if len(groups) >= 2:
            highest = max(groups, key=lambda item: item[1])
            lowest = min(groups, key=lambda item: item[1])
            if highest[1] != lowest[1]:
                sentences.append(
                    f"{comparison.get('metric', 'The measure')} is highest for {highest[0]} "
                    f"({_display(highest[1])}) and lowest for {lowest[0]} ({_display(lowest[1])}); "
                    f"the comparison is descriptive and does not establish cause."
                )
    elif categories and categories[0].get("values"):
        top = categories[0]["values"][0]
        sentences.append(
            f"The most common {categories[0]['label']} value is {top['value']}, appearing in {top['count']:,} records."
        )

    if quality_notes:
        note = quality_notes[0]
        if "at least 20%" in note:
            sentences.append(f"Review missingness in the flagged fields before drawing conclusions: {note}")
        elif "Some fields contain missing values" in note:
            sentences.append("Some missing values were reported, although no field crosses the 20% review threshold.")
        elif "No missing values" in note:
            sentences.append("No missing values were reported in the imported dataset profile.")
    return " ".join(sentences)


def _chart_interpretation(chart: dict[str, Any]) -> str:
    """Describe the observed pattern in a chart using only its computed values."""
    if chart.get("kind") == "heatmap":
        return "The matrix displays pairwise Pearson correlation coefficients; values close to +1.0 or -1.0 indicate strong association."

    labels = chart.get("labels", [])
    values = [value for value in chart.get("values", []) if _is_number(value)]
    if not labels or not values:
        return "The plot does not contain enough numeric observations for a data-specific finding."

    if chart.get("kind") == "missing_bar":
        max_idx = max(range(len(values)), key=lambda index: values[index])
        return f"{labels[max_idx]} has the highest missing proportion at {values[max_idx]:.1f}%."

    if chart.get("kind") == "histogram":
        peak_index = max(range(len(values)), key=lambda index: values[index])
        total = max(sum(values), 1)
        return (
            f"The most populated range is {labels[peak_index]}, containing {int(values[peak_index]):,} records "
            f"({values[peak_index] / total * 100:.1f}% of the plotted sample)."
        )

    if chart.get("kind") == "line":
        peak_index = max(range(len(values)), key=lambda index: values[index])
        low_index = min(range(len(values)), key=lambda index: values[index])
        direction = "increased" if values[-1] > values[0] else "decreased" if values[-1] < values[0] else "was broadly unchanged"
        return (
            f"The plotted measure {direction} from {_display(values[0])} in {labels[0]} to "
            f"{_display(values[-1])} in {labels[-1]}; its high point was {labels[peak_index]} "
            f"({_display(values[peak_index])}) and its low point was {labels[low_index]} ({_display(values[low_index])})."
        )

    if chart.get("group_counts"):
        if not values:
            return "No group records are available for comparison."
        max_val = max(values)
        min_val = min(values)
        diff = abs(max_val - min_val)

        # Detect equal or statistically indistinguishable group averages (Defect C fix)
        if diff < 1e-5 or (max_val != 0 and diff / abs(max_val) < 0.0001):
            return (
                f"All {len(labels)} plotted groups have virtually uniform averages at {_display(values[0])}."
            )

        peak_index = max(range(len(values)), key=lambda index: values[index])
        low_candidates = [i for i in range(len(values)) if i != peak_index]
        if not low_candidates:
            return f"{labels[peak_index]} is the only plotted group with an average of {_display(values[peak_index])}."
        low_index = min(low_candidates, key=lambda index: values[index])

        return (
            f"{labels[peak_index]} has the highest plotted average at {_display(values[peak_index])}, while "
            f"{labels[low_index]} is lowest at {_display(values[low_index])}; the difference is "
            f"{_display(diff)}."
        )

    peak_index = max(range(len(values)), key=lambda index: values[index])
    total = max(sum(values), 1)
    return (
        f"{labels[peak_index]} is the most frequent plotted category with {int(values[peak_index]):,} records "
        f"({values[peak_index] / total * 100:.1f}% of the displayed total)."
    )


def write_dataset_report_pdf(profile: dict[str, Any], title: str, path: Path) -> Path:
    """Create a structured, size-aware dataset report with relevant charts."""
    title = title.strip()[:25]
    if not title:
        raise ValueError("Enter a report title")
    path.parent.mkdir(parents=True, exist_ok=True)
    regular_font, bold_font = _register_fonts()

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        rightMargin=0.62 * inch, leftMargin=0.62 * inch,
        topMargin=0.62 * inch, bottomMargin=0.7 * inch,
        title=title, author="Quantai AI",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DatasetReportTitle", parent=styles["Title"], fontName=bold_font,
        fontSize=25, leading=30, textColor=colors.HexColor("#312e81"), alignment=TA_LEFT,
        spaceAfter=10,
    )
    subtitle_style = ParagraphStyle(
        "DatasetReportSubtitle", parent=styles["Normal"], fontName=regular_font,
        fontSize=10, leading=15, textColor=colors.HexColor("#64748b"), spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "DatasetReportSection", parent=styles["Heading1"], fontName=bold_font,
        fontSize=17, leading=21, textColor=colors.HexColor("#312e81"), spaceAfter=7,
    )
    chart_heading_style = ParagraphStyle(
        "DatasetReportChartHeading", parent=section_style,
        fontSize=13, leading=16, spaceBefore=2, spaceAfter=3,
    )
    plot_title_style = ParagraphStyle(
        "DatasetReportPlotTitle", parent=styles["Heading3"], fontName=bold_font,
        fontSize=10.5, leading=13, textColor=colors.HexColor("#334155"),
        spaceBefore=0, spaceAfter=4,
    )
    subhead_style = ParagraphStyle(
        "DatasetReportSubhead", parent=styles["Heading3"], fontName=bold_font,
        fontSize=10, leading=13, textColor=colors.HexColor("#334155"), spaceBefore=7,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "DatasetReportBody", parent=styles["BodyText"], fontName=regular_font,
        fontSize=9.3, leading=14, textColor=colors.HexColor("#334155"), spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "DatasetReportSmall", parent=body_style, fontSize=8, leading=11,
        textColor=colors.HexColor("#64748b"),
    )

    scale = profile.get("scale", "small")
    large = scale == "large"
    story: list[Any] = [
        Paragraph(_xml(title), title_style),
        Paragraph(f"Dataset intelligence report &nbsp;·&nbsp; {_xml(profile.get('name', 'Dataset'))}", subtitle_style),
        Paragraph("Executive summary", section_style),
        Paragraph(_xml(profile.get("summary", "Dataset profile")), body_style),
        _metric_cards(profile, body_style, bold_font),
        Spacer(1, 12),
        Paragraph("Key findings", subhead_style),
    ]
    for finding in profile.get("highlights", [])[:5]:
        story.append(Paragraph(f"• &nbsp; {_xml(finding)}", body_style))

    story.extend((Spacer(1, 6), Paragraph("Report scope", subhead_style)))
    story.append(Paragraph(
        "Use the executive summary for the main findings, the field guide for dataset scope and quality, then the measure and category sections for supporting evidence. "
        "Each plot has a clear title, a plain-language caption, and an interpretation. "
        "Sample-based plots state their row count; likely sensitive fields are omitted.",
        body_style,
    ))
    if not large:
        for note in profile.get("metric_notes", []):
            story.append(Paragraph(f"Interpretation note: {_xml(note)}", small_style))

    if large:
        story.extend((PageBreak(), _section("Dataset structure and quality", section_style)))
        story.append(Paragraph(
            f"The source has {profile.get('row_count', 0):,} records and {profile.get('col_count', 0):,} fields. "
            "The field guide below focuses on non-sensitive columns.", body_style,
        ))
        story.append(_catalog_table(profile, regular_font, bold_font))
        story.append(Paragraph("Data quality review", subhead_style))
        for note in profile.get("quality_notes", []):
            story.append(Paragraph(f"• &nbsp; {_xml(note)}", body_style))
        for note in profile.get("metric_notes", []):
            story.append(Paragraph(f"- {_xml(note)}", body_style))
        story.extend((PageBreak(), _section("Numeric measures", section_style)))
        story.append(Paragraph(
            "Averages summarize typical values, minimum and maximum show the observed range, and standard deviation describes spread. "
            "Missing percentages show how much data is unavailable for each measure.", body_style,
        ))
        story.append(_measure_table(profile, regular_font, bold_font))
        story.append(Paragraph("Measure interpretation", subhead_style))
        for note in _measure_insights(profile):
            story.append(Paragraph(f"- {_xml(note)}", body_style))
        distribution = profile.get("distribution_summary")
        if distribution:
            story.append(Paragraph("Distribution snapshot", subhead_style))
            story.append(Paragraph(
                f"Quartiles below summarize a bounded sample of {distribution['sample_count']:,} records for { _xml(distribution['metric']) }. "
                "The middle half of sampled values lies between the 25th and 75th percentiles.", body_style,
            ))
            snapshot_rows = [["Measure", "Sample rows", "25th percentile", "Median", "75th percentile", "Middle spread"]]
            snapshot_rows.append([
                _cell(distribution["metric"], bold_font),
                _cell(f"{distribution['sample_count']:,}", regular_font),
                _cell(_display(distribution.get("q1")), regular_font),
                _cell(_display(distribution.get("median")), regular_font),
                _cell(_display(distribution.get("q3")), regular_font),
                _cell(_display(distribution.get("iqr")), regular_font),
            ])
            story.append(_styled_table(snapshot_rows, [1.45, 0.9, 1.18, 0.75, 1.18, 1.1]))
            story.append(Paragraph(
                "Use the quartiles with the full-dataset minimum, maximum, and average above. The sample describes the distribution; "
                "it does not replace the full-dataset measures.", small_style,
            ))

        # Keep the numeric and categorical evidence in one continuous flow.
        # A hard page break here left large unused areas on both pages for
        # common datasets with only a few numeric measures and categories.
        story.append(_section("Category patterns", section_style))
        story.append(Paragraph(
            "Counts show how often each label occurs and whether a comparison is supported by many records. "
            "Small groups can have less stable averages; a difference between groups does not establish a cause.", body_style,
        ))
        story.append(_category_breakdown_table(profile, regular_font, bold_font))
        group_table = _group_comparison_table(profile, regular_font, bold_font, profile.get("row_count", 0))
        if group_table:
            story.append(Paragraph("Average by group", subhead_style))
            story.append(group_table)
        # The category and group tables already show counts, shares, and
        # support. Avoid repeating those rows in a separate section that
        # forced a mostly empty page on otherwise compact reports.
        _append_large_visual_pages(story, profile, section_style, chart_heading_style, plot_title_style, body_style, regular_font, small_style)
        if len(profile.get("charts", [])) >= 2 or len(profile.get("measures", [])) >= 3:
            story.extend((PageBreak(), _section("Analytical interpretation", section_style)))
            story.append(Paragraph(
                "This section translates the visuals into observations and decision questions. "
                "The findings describe patterns in the data; they do not establish causation.", body_style,
            ))
            for chart in profile.get("charts", []):
                story.append(Paragraph(_xml(chart.get("title", "Dataset plot")), subhead_style))
                purpose, use = _chart_context(chart)
                story.append(Paragraph(f"<b>Purpose:</b> {_xml(purpose)}", body_style))
                story.append(Paragraph(f"<b>Useful for:</b> {_xml(use)}", body_style))
                story.append(Paragraph(f"<b>Observed finding:</b> {_xml(chart.get('interpretation', 'No data-specific finding was available.'))}", body_style))
        if profile.get("col_count", 0) * profile.get("row_count", 0) >= 5_000_000 or profile.get("col_count", 0) >= 60:
            story.extend((PageBreak(), _section("Reading this report", section_style)))
            story.append(Paragraph(
                "This report is a structured profile of the uploaded dataset. It describes distributions and group comparisons, "
                "not causal effects. Confirm business context and investigate data-quality flags before making decisions.", body_style,
            ))
            story.append(Paragraph("Recommended follow-up", subhead_style))
            for item in _next_steps(profile):
                story.append(Paragraph(f"• &nbsp; {_xml(item)}", body_style))
    else:
        story.extend((PageBreak(), _section("Fields and data quality", section_style)))
        story.append(_catalog_table(profile, regular_font, bold_font))
        for note in profile.get("quality_notes", []):
            story.append(Paragraph(f"• &nbsp; {_xml(note)}", body_style))
        story.append(Paragraph("Measures and common groups", subhead_style))
        story.append(_measure_table(profile, regular_font, bold_font))
        story.append(_category_table(profile, regular_font, bold_font))
        if profile.get("charts"):
            story.extend((PageBreak(), _section("Plots and visual findings", section_style)))
            for chart in profile["charts"][:2]:
                story.extend(_plot_report_flowables(
                    chart, chart_heading_style, plot_title_style, body_style, regular_font, small_style,
                    image_height=3.1,
                ))
        else:
            story.append(Paragraph("No suitable non-sensitive fields were available for a reliable chart.", body_style))
        story.append(Paragraph("Recommended follow-up", subhead_style))
        for item in _next_steps(profile):
            story.append(Paragraph(f"• &nbsp; {_xml(item)}", body_style))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#e2e8f0"))
        canvas.line(doc.leftMargin, 0.48 * inch, letter[0] - doc.rightMargin, 0.48 * inch)
        canvas.setFont(regular_font, 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(doc.leftMargin, 0.3 * inch, "Quantai AI")
        canvas.drawRightString(letter[0] - doc.rightMargin, 0.3 * inch, f"{document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def _append_large_visual_pages(story, profile, section_style, chart_heading_style, plot_title_style, body_style, regular_font, small_style) -> None:
    charts = profile.get("charts", [])
    if not charts:
        story.extend((PageBreak(), _section("Plots and visual findings", section_style)))
        story.append(Paragraph("No plot was added because the dataset has no suitable non-sensitive fields to compare.", body_style))
        story.extend((PageBreak(), _section("Recommended follow-up", section_style)))
        for item in _next_steps(profile):
            story.append(Paragraph(f"\u2022 &nbsp; {_xml(item)}", body_style))
        return

    for chart in charts[:3]:
        story.append(PageBreak())
        image_height = 3.25 if chart.get("kind") in ("histogram", "missing_bar") else 4.5
        story.extend(_plot_report_flowables(
            chart, chart_heading_style, plot_title_style, body_style, regular_font, small_style,
            image_height=image_height, include_detail_table=True,
        ))

    if len(charts) < 2:
        story.extend((PageBreak(), _section("Recommended follow-up", section_style)))
        for item in _next_steps(profile):
            story.append(Paragraph(f"\u2022 &nbsp; {_xml(item)}", body_style))


def _plot_report_flowables(
    chart, heading_style, title_style, body_style, regular_font, small_style,
    image_height: float, include_detail_table: bool = False,
):
    items = [
        _section("Plot Report", heading_style),
        Paragraph(_xml(chart.get("title", "Dataset plot")), title_style),
    ]
    if chart.get("caption"):
        items.append(Paragraph(_xml(chart["caption"]), small_style))
    purpose, use = _chart_context(chart)
    items.append(Paragraph(f"<b>Purpose:</b> {_xml(purpose)}", small_style))
    items.append(Paragraph(f"<b>Useful for:</b> {_xml(use)}", small_style))
    if chart.get("interpretation"):
        items.append(Paragraph(f"<b>What the data shows:</b> {_xml(chart['interpretation'])}", body_style))

    image_data = _draw_chart(chart)
    if image_data:
        image = Image(image_data, width=6.85 * inch, height=image_height * inch, kind="proportional")
        image.hAlign = "LEFT"
        items.extend((image, Spacer(1, 4)))
    else:
        items.append(Paragraph("Chart unavailable for this profile.", body_style))

    if include_detail_table:
        detail_table = _chart_detail_table(chart, regular_font, body_style)
        if detail_table:
            items.append(detail_table)
            items.append(Spacer(1, 4))

    if chart.get("kind") == "histogram":
        explanation = (
            "The bars show how many sampled records fall into each value range. Taller bars identify the most common ranges; "
            "gaps can indicate rare values. The pattern describes this sample and does not explain why the values occur."
        )
    elif chart.get("kind") == "line":
        explanation = (
            "Each point shows the selected measure for a time bucket. Follow the line to see when values rise or fall, "
            "then compare those periods with business events before drawing conclusions."
        )
    elif chart.get("group_counts"):
        sizes = "; ".join(
            f"{_xml(group['label'])}: {group['count']:,} records"
            for group in chart["group_counts"]
        )
        explanation = (
            "The bars compare the average shown for each group. "
            f"Group sizes are {sizes}. Read the averages alongside these counts; the comparison shows association, not cause."
        )
    elif chart.get("kind") == "bar":
        explanation = (
            "Each bar represents the record count for one category. The tallest bars are the most common values in the dataset; "
            "they do not indicate that those categories perform better."
        )
    else:
        explanation = (
            "Compare the plotted values to see how the measure differs across the displayed groups. "
            "These differences describe the dataset and should not be read as causal effects."
        )
    items.append(Paragraph(f"<b>How to read this plot:</b> {_xml(explanation)}", small_style))
    return [KeepTogether(items)]


def _chart_context(chart: dict[str, Any]) -> tuple[str, str]:
    """Return a plain-language purpose and practical use for a chart."""
    kind = chart.get("kind")
    if kind == "line":
        metric = chart.get("metric") or chart.get("title", "the measure")
        return (
            f"Show how {metric} changes across the available time buckets.",
            "Spot direction, peaks, troughs, seasonality, and periods that deserve operational review.",
        )
    if kind == "histogram":
        metric = chart.get("metric") or chart.get("title", "the measure")
        return (
            f"Show how {metric} is distributed across value ranges in the plotted sample.",
            "Assess concentration, spread, unusual ranges, and whether an average represents a typical record.",
        )
    if kind == "heatmap":
        return (
            "Display pairwise correlation coefficients between numeric fields.",
            "Identify strong positive or negative relationships to explore colinearity and associations.",
        )
    if kind == "missing_bar":
        return (
            "Display the proportion of missing data across affected fields.",
            "Determine data quality risks, imputation needs, or features needing review.",
        )
    if chart.get("group_counts"):
        return (
            f"Compare the average {chart.get('metric', 'measure')} across {chart.get('dimension', 'groups')}.",
            "Find stronger or weaker segments while checking whether each comparison has enough records.",
        )
    return (
        f"Show the composition of the dataset across the displayed {chart.get('dimension', 'categories')}.",
        "Identify dominant segments, concentration, and categories that may need separate investigation.",
    )


def _chart_detail_table(chart, regular_font: str, body_style):
    labels = chart.get("labels", [])
    values = chart.get("values", [])
    if not labels or len(labels) != len(values):
        return None

    if chart.get("kind") == "histogram":
        rows = [["Value range", "Sample records", "Share of sample"]]
        total = max(sum(int(value or 0) for value in values), 1)
        for label, value in zip(labels, values):
            count = int(value or 0)
            rows.append([
                _cell(label, regular_font),
                _cell(f"{count:,}", regular_font),
                _cell(f"{count / total * 100:.1f}%", regular_font),
            ])
        return _styled_table(rows, [3.3, 1.6, 1.95])

    if chart.get("group_counts"):
        counts = chart["group_counts"]
        total = max(sum(int(group.get("count", 0)) for group in counts), 1)
        rows = [["Group", "Average", "Records", "Share of groups"]]
        for label, value, group in zip(labels, values, counts):
            count = int(group.get("count", 0))
            rows.append([
                _cell(label, regular_font),
                _cell(_display(value), regular_font),
                _cell(f"{count:,}", regular_font),
                _cell(f"{count / total * 100:.1f}%", regular_font),
            ])
        return _styled_table(rows, [2.2, 1.3, 1.4, 1.95])

    if chart.get("kind") == "missing_bar":
        rows = [["Field", "Missing percentage"]]
        for label, value in zip(labels, values):
            rows.append([_cell(label, regular_font), _cell(f"{float(value):.1f}%", regular_font)])
        return _styled_table(rows, [4.8, 2.05])

    if chart.get("kind") == "bar":
        rows = [["Category", "Records"]]
        for label, value in zip(labels[:12], values[:12]):
            rows.append([_cell(label, regular_font), _cell(f"{int(value or 0):,}", regular_font)])
        return _styled_table(rows, [4.9, 1.95])
    return None


def _chart_flowables(chart: dict[str, Any], body_style, regular_font: str, include_title: bool = True):
    items = [Paragraph(_xml(chart.get("title", "Dataset plot")), body_style)] if include_title else []
    if chart.get("caption"):
        items.append(Paragraph(_xml(chart["caption"]), body_style))
    image_data = _draw_chart(chart)
    if image_data:
        image = Image(image_data, width=6.85 * inch, height=3.75 * inch, kind="proportional")
        image.hAlign = "LEFT"
        items.extend((image, Spacer(1, 8)))
    else:
        items.append(Paragraph("Chart unavailable for this profile.", body_style))
    return items


def _draw_chart(chart: dict[str, Any]) -> io.BytesIO | None:
    kind = chart.get("kind", "bar")
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if kind == "heatmap":
            matrix = chart.get("matrix")
            labels = chart.get("labels") or []
            if not matrix or not labels:
                return None
            mat = np.array(matrix, dtype=float)
            n = len(labels)
            fig_size = (max(7.5, n * 1.1), max(4.8, n * 0.85))
            figure = Figure(figsize=fig_size, dpi=130, facecolor="white")
            axis = figure.add_subplot(111)
            cax = axis.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            figure.colorbar(cax, ax=axis, fraction=0.046, pad=0.04)
            axis.set_xticks(np.arange(n))
            axis.set_yticks(np.arange(n))
            axis.set_xticklabels([str(l)[:16] for l in labels], rotation=25, ha="right", fontsize=8)
            axis.set_yticklabels([str(l)[:16] for l in labels], fontsize=8)
            for i in range(n):
                for j in range(n):
                    val = mat[i, j]
                    text_color = "white" if abs(val) > 0.55 else "black"
                    axis.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8, weight="bold")
            figure.tight_layout(pad=1.5)
            output = io.BytesIO()
            FigureCanvasAgg(figure).print_png(output)
            output.seek(0)
            return output

        if kind == "missing_bar":
            labels = chart.get("labels") or []
            values = chart.get("values") or []
            if not labels or len(labels) != len(values):
                return None
            figure = Figure(figsize=(8.5, 4.5), dpi=130, facecolor="white")
            axis = figure.add_subplot(111)
            indices = np.arange(len(labels))
            bars = axis.barh(indices, values, color="#ef4444", edgecolor="#ffffff", height=0.6)
            axis.set_yticks(indices)
            axis.set_yticklabels([str(l)[:24] for l in labels], fontsize=8, color="#334155")
            axis.tick_params(axis="x", labelsize=8, colors="#475569")
            axis.grid(axis="x", color="#e2e8f0", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#cbd5e1")
            axis.spines["bottom"].set_color("#cbd5e1")
            axis.set_xlabel("Missing Percentage (%)", fontsize=9, color="#475569")
            for bar, val in zip(bars, values):
                axis.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                          va="center", ha="left", fontsize=8, color="#334155", weight="bold")
            axis.set_xlim(0, max(max(values) * 1.15, 10))
            figure.tight_layout(pad=1.5)
            output = io.BytesIO()
            FigureCanvasAgg(figure).print_png(output)
            output.seek(0)
            return output

        labels = chart.get("labels") or []
        values = chart.get("values") or []
        if not labels or len(labels) != len(values):
            return None

        figure_size = (8.5, 5.3) if kind in ("bar", "line") else (9, 4.7)
        figure = Figure(figsize=figure_size, dpi=130, facecolor="white")
        axis = figure.add_subplot(111)
        color = "#4f46e5"
        indices = np.arange(len(labels))
        if kind == "line":
            axis.plot(indices, values, color=color, linewidth=2.5, marker="o", markersize=4)
            axis.fill_between(indices, values, color=color, alpha=0.12)
        else:
            axis.bar(indices, values, color=color, edgecolor="#ffffff", linewidth=0.8)
        axis.set_xticks(indices)
        axis.set_xticklabels([str(label)[:24] for label in labels], rotation=25, ha="right", fontsize=8)
        axis.tick_params(axis="y", labelsize=8, colors="#475569")
        axis.tick_params(axis="x", colors="#475569")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#cbd5e1")
        axis.spines["bottom"].set_color("#cbd5e1")
        default_y_label = "Average value" if kind in ("line", "bar") else "Record count"
        axis.set_ylabel(chart.get("y_label", default_y_label), fontsize=9, color="#475569")
        figure.tight_layout(pad=1.5)
        output = io.BytesIO()
        FigureCanvasAgg(figure).print_png(output)
        output.seek(0)
        return output
    except Exception:
        log.exception("Could not draw dataset report chart")
        return None


def _catalog_table(profile, regular_font: str, bold_font: str):
    rows = [["Field", "Type", "Missing", "Distinct values", "Observed range"]]
    for column in profile.get("catalog", [])[:18]:
        range_text = "—"
        if column.get("minimum") is not None or column.get("maximum") is not None:
            range_text = f"{column.get('minimum') or '—'} to {column.get('maximum') or '—'}"
        rows.append([
            _cell(column.get("label", column.get("name", "")), bold_font),
            _cell(column.get("type", ""), regular_font),
            _cell(f"{column.get('null_pct', 0):.1f}%", regular_font),
            _cell(f"{column.get('distinct_count', 0):,}", regular_font),
            _cell(range_text, regular_font),
        ])
    if len(profile.get("catalog", [])) > 18:
        rows.append([_cell(f"+ {len(profile['catalog']) - 18} more fields", regular_font), "", "", "", ""])
    return _styled_table(rows, [1.55, 0.78, 0.72, 1.0, 2.7])


def _measure_table(profile, regular_font: str, bold_font: str):
    rows = [["Measure", "Average", "Minimum", "Maximum", "Std. dev.", "Missing"]]
    for measure in profile.get("measures", [])[:10]:
        rows.append([
            _cell(measure["label"], bold_font),
            _cell(_display(measure.get("average")), regular_font),
            _cell(_display(measure.get("minimum")), regular_font),
            _cell(_display(measure.get("maximum")), regular_font),
            _cell(_display(measure.get("stddev")), regular_font),
            _cell(f"{measure.get('null_pct', 0):.1f}%", regular_font),
        ])
    if len(rows) == 1:
        return Paragraph("No eligible numeric measures were found.", getSampleStyleSheet()["BodyText"])
    return _styled_table(rows, [1.45, 1.03, 1.03, 1.03, 1.03, 1.18])


def _category_table(profile, regular_font: str, bold_font: str):
    rows = [["Category", "Most common values", "Record counts"]]
    for category in profile.get("categories", [])[:4]:
        values = category.get("values", [])[:5]
        labels = ", ".join(str(item["value"])[:32] for item in values) or "—"
        counts = ", ".join(f"{item['count']:,}" for item in values) or "—"
        rows.append([_cell(category["label"], bold_font), _cell(labels, regular_font), _cell(counts, regular_font)])
    if len(rows) == 1:
        return Paragraph("No low-cardinality category fields were available for a stable breakdown.", getSampleStyleSheet()["BodyText"])
    return _styled_table(rows, [1.45, 3.75, 1.55])


def _category_breakdown_table(profile, regular_font: str, bold_font: str):
    rows = [["Field", "Value", "Records", "Share of dataset"]]
    total_rows = max(int(profile.get("row_count", 0)), 1)
    for category in profile.get("categories", [])[:3]:
        for item in category.get("values", [])[:8]:
            count = int(item.get("count", 0))
            rows.append([
                _cell(category["label"], bold_font),
                _cell(item.get("value", "Unknown"), regular_font),
                _cell(f"{count:,}", regular_font),
                _cell(f"{count / total_rows * 100:.1f}%", regular_font),
            ])
    if len(rows) == 1:
        return Paragraph("No low-cardinality category fields were available for a stable breakdown.", getSampleStyleSheet()["BodyText"])
    return _styled_table(rows, [1.85, 2.35, 1.15, 1.4])


def _group_comparison_table(profile, regular_font: str, bold_font: str, total_rows: int):
    chart = next((item for item in profile.get("charts", []) if item.get("group_counts")), None)
    if not chart:
        return None
    group_counts = chart["group_counts"]
    values = chart.get("values", [])
    total_group_rows = max(sum(int(item.get("count", 0)) for item in group_counts), 1)
    overall_average = chart.get("overall_average")
    rows = [["Group", "Average", "Records", "Share of rows", "Difference vs overall"]]
    for label, value, group in zip(chart.get("labels", []), values, group_counts):
        count = int(group.get("count", 0))
        difference = _number(value - overall_average) if overall_average is not None else None
        rows.append([
            _cell(label, bold_font),
            _cell(_display(value), regular_font),
            _cell(f"{count:,}", regular_font),
            _cell(f"{count / total_group_rows * 100:.1f}%", regular_font),
            _cell(_display(difference), regular_font),
        ])
    if len(rows) == 1:
        return None
    return _styled_table(rows, [1.5, 0.9, 1.0, 1.2, 1.8])


def _metric_cards(profile, body_style, bold_font: str):
    rows = [[
        Paragraph(f"<b>{profile.get('row_count', 0):,}</b><br/>Rows", body_style),
        Paragraph(f"<b>{profile.get('col_count', 0):,}</b><br/>Columns", body_style),
        Paragraph(f"<b>{_xml(profile.get('scale', 'small').title())}</b><br/>Report depth", body_style),
        Paragraph(f"<b>{profile.get('suggested_pages', 2)}</b><br/>Pages planned", body_style),
    ]]
    table = Table(rows, colWidths=[1.52 * inch] * 4)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#c7d2fe")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    return table


def _styled_table(rows, widths):
    data = []
    for row_index, row in enumerate(rows):
        if row_index == 0:
            data.append([Paragraph(f"<b>{_xml(cell)}</b>", getSampleStyleSheet()["BodyText"]) for cell in row])
        else:
            data.append(row)
    table = Table(data, colWidths=[width * inch for width in widths], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#312e81")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dbe2ea")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _section(title: str, style):
    return Paragraph(_xml(title), style)


def _cell(value: Any, font_name: str) -> Paragraph:
    style = ParagraphStyle(
        f"Cell_{font_name}", fontName=font_name, fontSize=7.5, leading=9.5,
        textColor=colors.HexColor("#334155"), splitLongWords=True,
    )
    return Paragraph(_xml(value), style)


def _next_steps(profile: dict[str, Any]) -> list[str]:
    steps = ["Compare the strongest groups with business targets and operational context."]
    if profile.get("measures"):
        steps.append(f"Ask follow-up questions about the trend and range of {profile['measures'][0]['label']}.")
    if profile.get("categories"):
        steps.append(f"Compare {profile['categories'][0]['label']} groups and inspect low-volume categories before drawing conclusions.")
    if any("missing" in note.lower() and "no missing" not in note.lower() for note in profile.get("quality_notes", [])):
        steps.append("Review fields with missing values before using them in a decision or model.")
    return steps[:4]


def _measure_insights(profile: dict[str, Any]) -> list[str]:
    notes = []
    for measure in profile.get("measures", []):
        name = measure["name"].lower()
        minimum = _display(measure.get("minimum"))
        maximum = _display(measure.get("maximum"))
        average = measure.get("average")
        deviation = measure.get("stddev")
        if "range" in name or "level" in name or "tier" in name:
            notes.append(
                f"{measure['label']} is an ordered level from {minimum} to {maximum}; treat its mean as a rough indicator rather than a continuous measurement."
            )
        elif average is not None and deviation is not None and abs(average) > 0 and deviation / abs(average) >= 2:
            notes.append(
                f"{measure['label']} varies widely (standard deviation {_display(deviation)} around an average of {_display(average)}); the average may not represent a typical record."
            )
        elif average is not None:
            notes.append(
                f"{measure['label']} averages {_display(average)} across observed values from {minimum} to {maximum}."
            )
    return notes[:4] or ["Review each measure's range and missingness before comparing groups or setting targets."]


def _category_insights(profile: dict[str, Any]) -> list[str]:
    total_rows = max(int(profile.get("row_count", 0)), 1)
    notes = []
    for category in profile.get("categories", [])[:4]:
        values = category.get("values", [])
        if not values:
            continue
        top = values[0]
        share = top["count"] / total_rows * 100
        note = f"{top['value']} is the most common {category['label']} value ({top['count']:,} records, {share:.1f}% of all rows)."
        if len(values) > 1:
            next_value = values[1]
            next_share = next_value["count"] / total_rows * 100
            note += f" {next_value['value']} appears in {next_value['count']:,} rows ({next_share:.1f}%)."
            if next_share < 5:
                note += " This small group has limited support, so treat its average as less stable."
        else:
            note += " No second category value was common enough for a reliable comparison."
        notes.append(note)
    if not notes:
        return ["No stable low-cardinality groups were available for a category breakdown."]
    return notes


def _is_sensitive(column: dict[str, Any]) -> bool:
    name = column["name"].lower()
    return bool(column.get("is_pii")) or any(hint in name for hint in _PII_HINTS)


def _metric_priority(column: dict[str, Any]) -> int:
    """Rank numeric columns by analytical usefulness — domain-agnostic."""
    name = column["name"].lower()
    # High-value business metrics (common across domains)
    if any(token in name for token in ("revenue", "sales", "profit", "income", "amount", "total")):
        return 0
    # Scores and ratings (present in many domains)
    if any(token in name for token in ("score", "rating", "grade", "rank")):
        return 1
    # Counts and quantities
    if any(token in name for token in ("votes", "quantity", "count", "volume", "weight")):
        return 2
    if _is_money_metric(name):
        return 3
    # Low-cardinality numerics are often encoded categories
    if column.get("distinct_count", 0) <= 10:
        return 5
    return 4


def _is_money_metric(name: str) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in ("range", "level", "tier")):
        return False
    return any(token in lowered for token in ("cost", "amount", "revenue", "sales", "profit", "price"))


def _category_priority(column: dict[str, Any]) -> int:
    """Rank categorical columns by analytical usefulness — domain-agnostic."""
    name = column["name"].lower()
    # Boolean-like flags are always useful dimensions
    if name.startswith(("has_", "is_", "can_")):
        return 0
    # General categorical dimensions common across domains
    if any(word in name for word in ("category", "type", "status", "segment", "group",
                                      "department", "class", "level", "tier", "channel",
                                      "gender", "grade", "brand")):
        return 1
    # Geographic and code dimensions — useful but often high-cardinality
    if any(word in name for word in ("currency", "country", "_code")):
        return 8
    if column.get("distinct_count", 0) > 30:
        return 6
    return 3


def _usable_group_dimension(name: str) -> bool:
    """Decide if a categorical column is suitable as a group-by dimension."""
    lowered = name.lower()
    return not any(token in lowered for token in ("currency", "country", "_code", "_id"))


def _is_key(column: dict[str, Any]) -> bool:
    name = column["name"].lower()
    return bool(column.get("is_key")) or bool(re.search(r"(^id$|_id$|^id_|(^|_)key($|_)|(^|_)code($|_)|uuid|serial)", name))


def _is_temporal_column(column: dict[str, Any]) -> bool:
    """Determine whether a column represents a time dimension (date, year, period)."""
    if column.get("is_date") or column.get("is_date_col"):
        return True
    col_type = column.get("type") or column.get("col_type", "")
    if col_type in ("date", "datetime"):
        return True
    if column.get("unit") == "year":
        return True
    name = column.get("name", "").lower()
    if name in ("year", "yr", "date", "datetime", "timestamp", "period", "quarter", "month", "day", "week"):
        return True
    if "year" in name:
        return True
    col_min = column.get("minimum")
    col_max = column.get("maximum")
    if col_min is not None and col_max is not None:
        try:
            min_f = float(col_min)
            max_f = float(col_max)
            if 1800 <= min_f <= 2090 and 1800 <= max_f <= 2100 and min_f.is_integer() and max_f.is_integer() and column.get("distinct_count", 0) <= 250:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _is_analytical_measure(column: dict[str, Any]) -> bool:
    """Determine whether a column is a genuine quantitative analytical measure."""
    col_type = column.get("type") or column.get("col_type", "")
    if col_type not in ("int", "float"):
        return False
    if _is_key(column):
        return False
    if _is_temporal_column(column):
        return False
    if "distinct_count" in column and column["distinct_count"] <= 1:
        return False
    return True


def _format_bin_range(edge_start: float, edge_end: float) -> str:
    """Format histogram bin edges cleanly without scientific notation."""
    def _fmt(val: float) -> str:
        abs_v = abs(val)
        if abs_v >= 1_000_000_000:
            return f"{val / 1_000_000_000:.2f}B"
        if abs_v >= 1_000_000:
            return f"{val / 1_000_000:.2f}M"
        if abs_v >= 1_000:
            return f"{val:,.0f}" if val.is_integer() else f"{val:,.1f}"
        if val.is_integer():
            return f"{int(val):,}"
        return f"{val:.2f}"
    return f"{_fmt(edge_start)}–{_fmt(edge_end)}"


def _quote(identifier: str) -> str:
    if not _SAFE_IDENTIFIER.fullmatch(identifier):
        raise ValueError("Unsafe dataset identifier")
    return f"`{identifier}`"


def _number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, 4) if math.isfinite(number) else None


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _label(value: Any) -> str:
    if value is None:
        return "Unknown"
    if isinstance(value, datetime):
        return value.strftime("%Y-%m")
    return str(value)[:100]


def _display(value: Any) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
        if not math.isfinite(number):
            return "—"
        return f"{number:,.2f}" if not number.is_integer() else f"{int(number):,}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _file_size(file_path: str | None) -> int:
    if not file_path:
        return 0
    try:
        return Path(file_path).stat().st_size
    except OSError:
        return 0


def _xml(value: Any) -> str:
    return escape(str(value)).replace("\r\n", "\n").replace("\n", "<br/>")


def _register_fonts() -> tuple[str, str]:
    if "QuantaiReport" not in pdfmetrics.getRegisteredFontNames():
        import reportlab

        font_dir = Path(reportlab.__file__).resolve().parent / "fonts"
        regular = font_dir / "Vera.ttf"
        bold = font_dir / "VeraBd.ttf"
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("QuantaiReport", str(regular)))
            pdfmetrics.registerFont(TTFont("QuantaiReport-Bold", str(bold)))
            pdfmetrics.registerFontFamily(
                "QuantaiReport", normal="QuantaiReport", bold="QuantaiReport-Bold",
                italic="QuantaiReport", boldItalic="QuantaiReport-Bold",
            )
            return "QuantaiReport", "QuantaiReport-Bold"
    return "Helvetica", "Helvetica-Bold"
