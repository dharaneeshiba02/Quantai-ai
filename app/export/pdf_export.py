"""Generate a printable PDF report for a completed dataset query."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PDF_MAX_ROWS = 500
PDF_MAX_COLUMNS = 10


async def export_pdf(df: pd.DataFrame, run_meta: dict[str, Any], path: Path) -> Path:
    """Write a valid PDF containing the answer and a readable result preview."""
    path.parent.mkdir(parents=True, exist_ok=True)

    visible_columns = list(df.columns[:PDF_MAX_COLUMNS])
    visible_df = df.loc[:, visible_columns].head(PDF_MAX_ROWS) if visible_columns else df.head(0)
    page_size = landscape(letter) if len(visible_columns) > 4 else letter
    doc = SimpleDocTemplate(
        str(path),
        pagesize=page_size,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.65 * inch,
        title="Quantai AI Analysis Report",
        author="Quantai AI",
    )

    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "QuantaiTitle", parent=base_styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=24, textColor=colors.HexColor("#312e81"), alignment=TA_LEFT,
        spaceAfter=8,
    )
    label_style = ParagraphStyle(
        "QuantaiLabel", parent=base_styles["Heading3"], fontName="Helvetica-Bold",
        fontSize=10, leading=13, textColor=colors.HexColor("#334155"), spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "QuantaiBody", parent=base_styles["BodyText"], fontName="Helvetica",
        fontSize=9.5, leading=14, textColor=colors.HexColor("#1e293b"),
        splitLongWords=True, spaceAfter=5,
    )
    cell_style = ParagraphStyle(
        "QuantaiCell", parent=body_style, fontSize=7.5, leading=9,
        textColor=colors.HexColor("#1e293b"), spaceAfter=0,
    )
    header_style = ParagraphStyle(
        "QuantaiTableHeader", parent=cell_style, fontName="Helvetica-Bold",
        textColor=colors.white,
    )

    story = [
        Paragraph("Quantai AI Analysis Report", title_style),
        Paragraph(f"Generated {datetime.now().strftime('%B %d, %Y at %H:%M')}", base_styles["Italic"]),
        Spacer(1, 10),
        Paragraph("Question", label_style),
        Paragraph(_paragraph_text(run_meta.get("question") or "Dataset analysis"), body_style),
    ]

    insight = str(run_meta.get("insight") or "").strip()
    if insight:
        story.extend((Paragraph("Summary", label_style), Paragraph(_paragraph_text(insight), body_style)))

    quality = []
    if run_meta.get("confidence"):
        quality.append(f"Confidence: {_paragraph_text(run_meta['confidence'])}")
    quality.append("Verified" if run_meta.get("verified") else "Not independently verified")
    story.extend((Paragraph("Analysis details", label_style), Paragraph(" &nbsp; | &nbsp; ".join(quality), body_style)))

    snapshot_rows = [[Paragraph("Data snapshot", header_style), Paragraph("Value", header_style)]]
    snapshot_rows.extend([
        [Paragraph("Rows returned", cell_style), Paragraph(f"{len(df):,}", cell_style)],
        [Paragraph("Columns returned", cell_style), Paragraph(f"{len(df.columns):,}", cell_style)],
        [Paragraph("Numeric measures", cell_style), Paragraph(f"{len(df.select_dtypes(include='number').columns):,}", cell_style)],
    ])
    snapshot = Table(snapshot_rows, colWidths=[2.2 * inch, 2.2 * inch], hAlign="LEFT")
    snapshot.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend((Paragraph("Data snapshot", label_style), snapshot))

    observations = _result_observations(df)
    if observations:
        story.append(Paragraph("Data observations", label_style))
        for observation in observations:
            story.append(Paragraph(f"- {_paragraph_text(observation)}", body_style))

    story.append(Paragraph("Query results", label_style))
    if visible_df.empty:
        story.append(Paragraph("No result rows were returned for this query.", body_style))
    else:
        table_data = [[Paragraph(_paragraph_text(column), header_style) for column in visible_columns]]
        for row in visible_df.itertuples(index=False, name=None):
            table_data.append([Paragraph(_paragraph_text(_cell_text(value)), cell_style) for value in row])

        available_width = page_size[0] - doc.leftMargin - doc.rightMargin
        widths = [available_width / len(visible_columns)] * len(visible_columns)
        results_table = LongTable(
            table_data,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
            splitByRow=1,
        )
        results_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(results_table)

        notes = []
        if len(df) > PDF_MAX_ROWS:
            notes.append(f"Showing the first {PDF_MAX_ROWS:,} of {len(df):,} result rows.")
        if len(df.columns) > PDF_MAX_COLUMNS:
            notes.append(f"Showing the first {PDF_MAX_COLUMNS} of {len(df.columns)} result columns.")
        if notes:
            story.extend((Spacer(1, 6), Paragraph(_paragraph_text(" ".join(notes)), base_styles["Italic"])))

    def draw_footer(canvas, _document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(doc.leftMargin, 0.45 * inch, page_size[0] - doc.rightMargin, 0.45 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(doc.leftMargin, 0.28 * inch, "Generated locally by Quantai AI")
        canvas.drawRightString(page_size[0] - doc.rightMargin, 0.28 * inch, f"Page {_document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_footer, onLaterPages=draw_footer)
    return path


def _paragraph_text(value: Any) -> str:
    """Escape untrusted text for ReportLab's small XML-like paragraph syntax."""
    return escape(str(value)).replace("\r\n", "\n").replace("\n", "<br/>")


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float,)) and pd.isna(value):
        return ""
    try:
        if pd.api.types.is_scalar(value) and pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _result_observations(df: pd.DataFrame) -> list[str]:
    """Create compact, deterministic observations for exported query results."""
    observations: list[str] = []
    for column in list(df.select_dtypes(include="number").columns)[:3]:
        series = df[column].dropna()
        if series.empty:
            continue
        observations.append(
            f"{column}: average {_format_number(series.mean())}, ranging from "
            f"{_format_number(series.min())} to {_format_number(series.max())}."
        )
    for column in list(df.select_dtypes(include=["object", "category", "bool"]).columns)[:3]:
        counts = df[column].dropna().value_counts()
        if counts.empty:
            continue
        top_value = counts.index[0]
        observations.append(
            f"{column}: the most common value is {_cell_text(top_value)} "
            f"({int(counts.iloc[0]):,} of {len(df):,} rows)."
        )
    return observations[:5]


def _format_number(value: Any) -> str:
    try:
        number = float(value)
        return f"{number:,.2f}" if number % 1 else f"{number:,.0f}"
    except (TypeError, ValueError):
        return _cell_text(value)
