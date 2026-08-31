"""Shared color tokens and a Plotly template registered as the app default.
Imported once at module load (from render.py) so every chart in the app
picks it up automatically - no per-figure overrides needed.
"""

import plotly.graph_objects as go
import plotly.io as pio

BG = "#0a0b0f"
SURFACE = "#13151b"
CARD = "#191c25"
CARD_HOVER = "#1f2330"
BORDER = "rgba(255,255,255,0.08)"
BORDER_STRONG = "rgba(255,255,255,0.14)"
TEXT = "#f0f1f5"
MUTED = "#8b8fa3"

ACCENT = "#7c6cf6"
ACCENT_SOFT = "rgba(124,108,246,0.16)"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
ROSE = "#fb7185"
SKY = "#38bdf8"

CATEGORICAL = [
    "#7c6cf6", "#2dd4bf", "#fbbf24", "#fb7185", "#38bdf8",
    "#a3e635", "#f472b6", "#94a3b8", "#fb923c", "#4ade80", "#c084fc",
]

GRADE_COLOR = {"A": "#4ade80", "B": "#2dd4bf", "C": "#fbbf24", "D": "#fb923c", "F": "#fb7185"}
SEV_COLOR = {"high": "#fb7185", "medium": "#fbbf24", "low": "#94a3b8"}
DIR_COLOR = {"above": "#fb7185", "below": "#2dd4bf", "neutral": "#8b8fa3"}

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

_template = go.layout.Template()
_template.layout = go.Layout(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(family=FONT_FAMILY, color=TEXT, size=12.5),
    colorway=CATEGORICAL,
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER_STRONG, linecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER_STRONG, linecolor=BORDER, tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
    margin=dict(t=52, b=40, l=52, r=24),
    title=dict(font=dict(size=14, color=TEXT)),
    hoverlabel=dict(bgcolor=CARD_HOVER, font=dict(color=TEXT, family=FONT_FAMILY), bordercolor=BORDER_STRONG),
    colorscale=dict(
        sequential=[[0, "#2a2140"], [0.5, ACCENT], [1, TEAL]],
        diverging=[[0, ROSE], [0.5, "#3a3d4d"], [1, TEAL]],
    ),
)
pio.templates["fintech_dark"] = _template
pio.templates.default = "fintech_dark"

TABLE_STYLE = dict(
    style_table={"overflowX": "auto", "borderRadius": "10px", "border": f"1px solid {BORDER}"},
    style_header={
        "backgroundColor": SURFACE, "color": MUTED, "fontWeight": 600, "fontSize": "11px",
        "textTransform": "uppercase", "letterSpacing": "0.04em", "border": "none",
        "borderBottom": f"1px solid {BORDER_STRONG}", "padding": "10px 14px",
    },
    style_cell={
        "backgroundColor": CARD, "color": TEXT, "textAlign": "left", "padding": "10px 14px",
        "fontSize": "13px", "border": "none", "borderBottom": f"1px solid {BORDER}",
        "fontFamily": FONT_FAMILY,
    },
    style_data_conditional=[
        {"if": {"row_index": "odd"}, "backgroundColor": SURFACE},
    ],
)
