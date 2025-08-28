import os, re, io, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc
import dash_bootstrap_components as dbc

PASS_CLR = "#3ED4AE"
FAIL_CLR = "#FF45FC"

def _pick_col(df, candidates):
    norm = lambda s: str(s).strip().lower().replace(" ", "").replace("_", "")
    m = {norm(c): c for c in df.columns}
    for cand in candidates:
        k = norm(cand)
        if k in m:
            return m[k]
        for kk, vv in m.items():
            if k in kk:
                return vv
    return None

# Data preparation
def _coerce_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce", format="mixed")

    if s.notna().sum() == 0:
        try_s = pd.to_datetime(series.astype(str).str.strip(), format="%b %Y", errors="coerce")
        s = try_s if try_s.notna().sum() > 0 else s

    if s.notna().sum() == 0:
        try_s = pd.to_datetime(series.astype(str).str.strip(), format="%Y-%m", errors="coerce")
        s = try_s if try_s.notna().sum() > 0 else s

    return s



def prepare_dataframe(df: pd.DataFrame):
    WAVE   = _pick_col(df, ["Wave","wave","wave_id","group_id"])
    COURSE = _pick_col(df, ["course_name","Course","course"])
    GENDER = _pick_col(df, ["Gender","gender"])
    AGE    = _pick_col(df, ["age","Age"])
    START  = _pick_col(df, ["start_date","Start Date","wave_start_date","cohort_start"])
    END    = _pick_col(df, ["end_date","End Date","wave_end_date","cohort_end"])
    EX1    = _pick_col(df, ["exam_1_score","exam1","exam_1"])
    EX2    = _pick_col(df, ["exam_2_score","exam2","exam_2"])
    EX3    = _pick_col(df, ["exam_3_score","exam3","exam_3"])
    BG     = _pick_col(df, ["background_level","background","level","student_level"])

    out = df.copy()

    if WAVE:
        out["_wave_num"] = out[WAVE].astype(str).str.extract(r"(\d+)", expand=False).astype("Int64")
    else:
        out["_wave_num"] = pd.Series([pd.NA]*len(out), dtype="Int64")

    if START and START in out.columns:
        out[START] = _coerce_datetime(out[START])
    if END and END in out.columns:
        out[END] = _coerce_datetime(out[END])

    if all(c is not None and c in out.columns for c in [EX1, EX2, EX3]):
        out["average_exam_score"] = out[[EX1, EX2, EX3]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    else:
        out["average_exam_score"] = pd.to_numeric(out.get("average_exam_score", np.nan), errors="coerce")

    COLS = dict(WAVE=WAVE, COURSE=COURSE, GENDER=GENDER, AGE=AGE,
                START=START, END=END, EX1=EX1, EX2=EX2, EX3=EX3, BG=BG)
    return out, COLS


def _load_default_raw_dataframe() -> pd.DataFrame:
    csv_path = os.path.join("data", "uploaded_latest.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    sample_names = [
        "synthetic_student_dataset_200 - Sheet1.csv",
        "synthetic_student_dataset_200.csv",
    ]
    for name in sample_names:
        if os.path.exists(name):
            return pd.read_csv(name)

    # 3) last resort: any CSV in /data
    if os.path.isdir("data"):
        for fn in os.listdir("data"):
            if fn.lower().endswith(".csv"):
                try:
                    return pd.read_csv(os.path.join("data", fn))
                except Exception:
                    pass
    return pd.DataFrame()


def load_df():
    snap_path = os.path.join("data", "last_payload.json")
    if os.path.exists(snap_path):
        with open(snap_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        df = pd.read_json(io.StringIO(saved["df"]), orient="split")
        cols = saved["cols"]
        return df, cols

    df_raw = _load_default_raw_dataframe()
    return prepare_dataframe(df_raw)


# Filters
def build_options(df: pd.DataFrame, COLS: dict):
    d = df.copy()
    wave_col   = COLS.get("WAVE")
    course_col = COLS.get("COURSE")
    start_col  = COLS.get("START")

    if start_col and start_col in d.columns:
        d[start_col] = _coerce_datetime(d[start_col])
    # waves
    if wave_col and wave_col in d.columns:
        wave_vals = sorted(d[wave_col].dropna().astype(str).unique().tolist())
    else:
        wave_vals = []
    wave_options = ["All"] + wave_vals
    # courses
    if course_col and course_col in d.columns:
        course_vals = sorted(d[course_col].dropna().astype(str).unique().tolist())
    else:
        course_vals = []
    course_options = ["All"] + course_vals
    # years
    if start_col and start_col in d.columns:
        years = d[start_col].dt.year.dropna().astype(int).unique().tolist()
        years = sorted(years)
        year_options = ["All"] + [str(y) for y in years]
    else:
        year_options = ["All"]

    return wave_options, course_options, year_options

def _as_list(v):
    if v is None: return []
    if isinstance(v, (list, tuple, set)): return list(v)
    return [v]

def _has_all(vals):
    return any(str(x).strip().lower() == "all" for x in vals)

def _casefold_set(vals):
    return {str(x).strip().casefold() for x in vals}

def apply_filters(df: pd.DataFrame, COLS: dict, wave_sel, status_sel, course_sel, year_sel, today):
    d = df.copy()

    WAVE_COL   = COLS.get("WAVE")
    COURSE_COL = COLS.get("COURSE")
    START_COL  = COLS.get("START")
    END_COL    = COLS.get("END")

    # coerce date columns (idempotent and safe)
    if START_COL and START_COL in d.columns:
        d[START_COL] = _coerce_datetime(d[START_COL])
    if END_COL and END_COL in d.columns:
        d[END_COL] = _coerce_datetime(d[END_COL])

    today = pd.Timestamp(today).normalize()

    # Wave
    waves = _as_list(wave_sel)
    if WAVE_COL and WAVE_COL in d.columns and waves and not _has_all(waves):
        sel = {str(x) for x in waves}
        d = d[d[WAVE_COL].astype(str).isin(sel)]
    # Course
    courses = _as_list(course_sel)
    if COURSE_COL and COURSE_COL in d.columns and courses and not _has_all(courses):
        sel = {str(x) for x in courses}
        d = d[d[COURSE_COL].astype(str).isin(sel)]
    # Year
    if year_sel and str(year_sel).lower() != "all":
        try:
            y = int(str(year_sel))
        except Exception:
            y = None
        if y is not None:
            if START_COL and START_COL in d.columns:
                d = d[d[START_COL].dt.year == y]
            elif "_start_year" in d.columns:
                yr = pd.to_numeric(d["_start_year"], errors="coerce").astype("Int64")
                d = d[yr == y]
    # Status
    if status_sel and str(status_sel).lower() != "all":
        key = str(status_sel).strip().lower()
        if key.startswith("active"):
            mask = pd.Series(True, index=d.index)
            if START_COL and START_COL in d.columns:
                mask &= d[START_COL].le(today)
            if END_COL and END_COL in d.columns:
                mask &= d[END_COL].isna() | d[END_COL].ge(today)
            d = d[mask]
        elif key.startswith("alumni"):
            if END_COL and END_COL in d.columns:
                d = d[d[END_COL].notna() & d[END_COL].lt(today)]
            else:
                d = d.iloc[0:0]

    return d


# Helpers
def store_payload(df, COLS):
    return {
        "df": df.to_json(orient="split", date_format="iso"),
        "cols": COLS
    }


def default_store_payload():
    df, COLS = load_df()
    return store_payload(df, COLS)


# UI builders
def make_filters(wave_options, course_options, year_options):
    def _dedupe(opts):
        norm = lambda s: str(s).strip().casefold()
        cleaned, seen = [], set()
        for o in (opts or []):
            if isinstance(o, dict):
                label = o.get("label", o.get("value", ""))
                value = o.get("value", o.get("label", ""))
            else:
                label = value = o
            k = norm(label)
            if k and k not in seen:
                cleaned.append({"label": label, "value": value})
                seen.add(k)
        return cleaned

    def _with_all_first(opts):
        opts = _dedupe(opts)
        opts = [o for o in opts if str(o["label"]).strip().casefold() != "all"]
        return [{"label": "All", "value": "All"}] + opts

    wave_opts   = _with_all_first(wave_options)
    course_opts = _with_all_first(course_options)
    year_opts   = _with_all_first(year_options)

    status_opts = _with_all_first([
        {"label": "All",    "value": "All"},
        {"label": "Active", "value": "Active"},
        {"label": "Alumni", "value": "Alumni"},
    ])

    reset_btn = html.Div(
        dbc.Button("Reset", id="reset-filters", n_clicks=0, class_name="reset-btn"),
        className="reset-wrap"
    )

    return dbc.Card(
        dbc.CardBody(
            [
                html.Label("Wave", className="fw-semibold mb-2"),
                dcc.Dropdown(id="filter-wave", options=wave_opts, value=["All"],
                             multi=True, clearable=False, persistence=True),
                html.Br(),

                html.Label("Year", className="fw-semibold mb-2"),
                dcc.Dropdown(id="filter-year", options=year_opts, value="All",
                             clearable=False, persistence=True),
                html.Br(),

                html.Label("Status", className="fw-semibold mb-2"),
                dcc.Dropdown(id="filter-status", options=status_opts, value="All",
                             clearable=False, persistence=True),
                html.Br(),

                html.Label("Course", className="fw-semibold mb-2"),
                dcc.Dropdown(id="filter-course", options=course_opts, value=["All"],
                             multi=True, clearable=False, persistence=True),

                html.Div(reset_btn, className="mt-4"),
            ]
        ),
        class_name="shadow-sm",
        style={
            "backgroundColor": "white",
            "border": "1px solid #e0e6f1",
            "borderRadius": "14px",
        },
    )


def kpi_card(icon_class: str, label: str, value_id: str):
    return dbc.Card(
        dbc.CardBody(
            dbc.Row([
                dbc.Col(
                    html.Div(
                        html.I(className=f"bi {icon_class}",
                               style={"fontSize": "1.4rem", "color": "#7E3FF2"}),
                        style={"width": "48px", "height": "48px", "borderRadius": "50%",
                               "backgroundColor": "#DCE7FF", "display": "flex",
                               "alignItems": "center", "justifyContent": "center"}
                    ),
                    width="auto", className="d-flex align-items-center"
                ),
                dbc.Col(
                    html.Div([
                        html.Div(label, style={"fontSize": "0.9rem", "fontWeight": 600, "color": "#46546B"}),
                        html.Div(id=value_id, style={"fontSize": "1.8rem", "fontWeight": 700,
                                                     "color": "#24314A", "lineHeight": "1.1"}),
                    ])
                ),
            ], class_name="g-3"),
            style={"padding": "20px 16px"}
        ),
        class_name="shadow-sm",
        style={"borderRadius": "14px", "backgroundColor": "#F5F8FF", "minHeight": "100px"}
    )


# Figures

def donut_figure(title: str, pct: float | None, color: str) -> go.Figure:
    fig = go.Figure()
    if pct is None or pd.isna(pct):
        fig.update_layout(title=title, title_x=0.5, title_font=dict(size=14),
                          margin=dict(l=36, r=36, t=56, b=8),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          showlegend=False)
        return fig
    pct = float(np.clip(pct, 0, 100))
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct], hole=0.65,
        marker=dict(colors=[color, "#e6e6e6"]),
        textinfo="none", showlegend=False, sort=False
    ))
    fig.add_annotation(text=f"{pct:.2f}%", x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#1C2436"))
    fig.update_layout(title=title, title_x=0.5, title_font=dict(size=14),
                      margin=dict(l=36, r=36, t=56, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gender_figure(d, GENDER):
    if not GENDER or d.empty:
        fig = px.pie()
        fig.update_layout(title_text="Student Count by Gender", title_x=0.5, height=260)
        return fig
    gender_counts = d[GENDER].astype(str).value_counts().rename_axis("Gender").reset_index(name="Count")
    fig = px.pie(gender_counts, names="Gender", values="Count", hole=0.6, color="Gender",
                 color_discrete_map={"M": "#8f57ef", "F": "#3ed4ae"})
    fig.update_traces(textinfo="value", textfont_size=13, hovertemplate="%{label}: %{value}<extra></extra>")
    fig.update_layout(title_text="Student Count by Gender", title_x=0.5,
                      margin=dict(l=10, r=10, t=40, b=10), height=260,
                      legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def age_figure(d, AGE):
    if not AGE or d.empty:
        fig = px.bar()
        fig.update_layout(title_text="Enrollment by Age Group", title_x=0.5, height=260)
        return fig
    ages = pd.to_numeric(d[AGE], errors="coerce").dropna().astype(int)
    if ages.empty:
        fig = px.bar()
        fig.update_layout(title_text="Enrollment by Age Group", title_x=0.5, height=260)
        return fig
    bins = [13, 16, 20, 25, 30, np.inf]
    labels = ["13-15", "16-20", "21-25", "26-30", "30+"]
    age_groups = pd.cut(ages, bins=bins, labels=labels, right=True, include_lowest=True)
    age_counts = age_groups.value_counts(sort=False).rename_axis("Age Group").reset_index(name="Students")
    fig = px.bar(age_counts, y="Age Group", x="Students", orientation="h")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, title="Age Group", showgrid=False)
    fig.update_traces(marker_color="#8f57ef", marker_line_width=0,
                      text=age_counts["Students"].astype(str),
                      texttemplate="%{text}", textposition="outside", cliponaxis=False)
    fig.update_layout(title_text="Enrollment by Age Group", title_x=0.5,
                      bargap=0.25, margin=dict(l=10, r=10, t=40, b=10), height=260,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Students")
    return fig


def passfail_figure(d, COURSE):
    if "average_exam_score" not in d.columns or d.empty or not COURSE or COURSE not in d.columns:
        fig = px.bar(title="Pass vs Fail Rate by Course")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    tmp = d.copy()
    tmp["result"] = np.where(pd.to_numeric(tmp["average_exam_score"], errors="coerce") >= 70, "Pass", "Fail")
    counts = tmp.groupby([COURSE, "result"], observed=True).size().reset_index(name="count")
    if counts.empty:
        fig = px.bar(title="Pass vs Fail Rate by Course")
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig
    counts["percent"] = counts.groupby(COURSE)["count"].transform(lambda x: 100 * x / x.sum())
    order = counts[counts["result"] == "Fail"].sort_values("percent", ascending=False)[COURSE]
    counts[COURSE] = pd.Categorical(counts[COURSE], categories=order, ordered=True)
    fig = px.bar(counts, x="percent", y=COURSE, color="result", orientation="h",
                 barmode="stack", text=counts["percent"].round(1),
                 labels={"percent":"%", "result":"", **{COURSE: "Course"}}, title="Pass vs Fail Rate by Course",
                 color_discrete_map={"Pass": PASS_CLR, "Fail": FAIL_CLR})
    fig.update_traces(textposition="inside", texttemplate="%{text:.1f}%")
    fig.update_layout(xaxis=dict(range=[0, 100], ticksuffix="%", title=None),
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig
