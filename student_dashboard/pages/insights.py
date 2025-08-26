import re, io, json
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from .overview_parts import load_df, apply_filters

dash.register_page(__name__, path="/insights", name="Insights", order=1)

DEFAULT_DF, DEFAULT_COLS = load_df()
TODAY = pd.Timestamp.today().normalize()

# Styling
def white_card(child, pad=15):
    return dbc.Card(
        dbc.CardBody(child, style={"padding": f"{pad}px"}),
        class_name="shadow-sm",
        style={
            "backgroundColor": "white",
            "border": "1px solid #e0e6f1",
            "borderRadius": "12px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
        },
    )

BLUE = "#3A72EF"
PURPLE = "#8F57EF"
CYAN = "#36E5FF"
FUCHSIA = "#FF45FC"

# Helpers
def _pick_col(df_, candidates):
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", str(s).lower())
    m = {norm(c): c for c in df_.columns}
    for cand in candidates:
        k = norm(cand)
        if k in m:
            return m[k]
        for kk, vv in m.items():
            if k in kk:
                return vv
    return None

WAVE_COL   = DEFAULT_COLS.get("WAVE")
COURSE_COL = DEFAULT_COLS.get("COURSE")
AGE_COL    = DEFAULT_COLS.get("AGE")
START_COL  = DEFAULT_COLS.get("START")
END_COL    = DEFAULT_COLS.get("END")

AVG_EXAM_COL = _pick_col(DEFAULT_DF, ["average_exam_score","avg_exam","exam_avg","exam_score_avg"])
STRESS_COL   = _pick_col(DEFAULT_DF, ["stress_level","stress","avg_stress","stressscore"])
FOCUS_COL    = _pick_col(DEFAULT_DF, ["focus_level","focus"])
SMBR_COL     = _pick_col(DEFAULT_DF, ["smoking_breaks_level","smoking_breaks","breaks_level"])
SLEEP_COL    = _pick_col(DEFAULT_DF, ["sleep_hours","sleep","sleep_hours_per_night","sleepdurationhours","sleep_duration"])
ATT_BIN_COL  = _pick_col(DEFAULT_DF, ["attendance_bin","lecture_attendance_bin"])
ATT_PCT_COL  = _pick_col(DEFAULT_DF, ["lecture_attendance","attendance_percent","attendance_pct","attendance_percentage"])
PARTIC_COL   = _pick_col(DEFAULT_DF, ["participation_rate","class_participation","participationpct",
                                      "participation","participation_percent","participation_percentage"])
VIDEO_COL    = _pick_col(DEFAULT_DF, ["video_lecture_engagement","video_engagement","video_engagement_rate",
                                      "video_engagement_percent","video_engagement_percentage","video_lecture_completion",
                                      "video_lectures_completion","video_watched_percent","video_watch_rate","video_watched_pct"])
VIDEO_LVL_COL   = _pick_col(DEFAULT_DF, ["video_lecture_engagement_level","video_level","video_engagement_level",
                                         "video_lectures_engagement_level"])
READING_COL     = _pick_col(DEFAULT_DF, ["assigned_reading_completion","reading_completion","reading_completion_rate",
                                         "reading_completion_percent","reading_completion_percentage","assigned_reading_completion_rate",
                                         "assigned_reading_completion_pct"])
READING_LVL_COL = _pick_col(DEFAULT_DF, ["assigned_readings_completion_level","assigned_reading_completion_level",
                                         "reading_completion_level","reading_level"])
SELF_STD_COL    = _pick_col(DEFAULT_DF, ["self_study_hours","selfstudy_hours","study_hours"])
CONTINUE_COL    = _pick_col(DEFAULT_DF, ["continued_after_failure","continued","continue_status"])
JOBUNI_COL      = _pick_col(DEFAULT_DF, ["job_uni_status","jobunistatus","job_uni","job and uni",
                                         "employment_study_status","study_job_status","jobuni"])
JOB_COL         = _pick_col(DEFAULT_DF, ["job","job_status","has_job","employment_status","employed",
                                         "work_status","working","jobflag"])
UNI_COL         = _pick_col(DEFAULT_DF, ["uni","university","is_student","student_status","enrolled",
                                         "enrollment_status","study_status","studying","uni_enrolled"])

# Layout
wave_compare_block = dbc.Card(
    dbc.CardBody(dcc.Graph(id="fig-wave-compare", config={"displayModeBar": False}, style={"height": "360px"})),
    class_name="shadow-sm",
    style={"backgroundColor": "white", "border": "1px solid #e0e6f1", "borderRadius": "12px"}
)

focus_x_options = [
    {"label": "Age Group",              "value": "age"},
    {"label": "Smoking & Breaks Level", "value": "smbr"},
    {"label": "Sleep Hours",            "value": "sleep"},
]
exam_x_options = [
    {"label": "Lecture Attendance",          "value": "attendance"},
    {"label": "Participation Rate",          "value": "participation"},
    {"label": "Video Lecture Engagement",    "value": "video"},
    {"label": "Self Study Hours",            "value": "selfstudy"},
    {"label": "Assigned Reading Completion", "value": "reading"},
]

controls = dbc.Row(
    [
        dbc.Col(
            white_card(html.Div([
                html.Label("Focus Level vs Feature", className="fw-semibold mb-2"),
                dcc.Dropdown(id="x-focus", options=focus_x_options, value="age", clearable=False),
            ]), pad=12),
            md=12, lg=6
        ),
        dbc.Col(
            white_card(html.Div([
                html.Label("Average Exam Score vs Feature", className="fw-semibold mb-2"),
                dcc.Dropdown(id="x-exam", options=exam_x_options, value="attendance", clearable=False),
            ]), pad=12),
            md=12, lg=6
        ),
    ],
    class_name="gx-4 gy-3 align-items-end",
    style={"marginTop": "18px", "marginBottom": "12px"}
)

jobuni_y_options = [
    {"label": "Average Stress Level",    "value": "stress"},
    {"label": "Focus Level",             "value": "focus"},
    {"label": "Self Study Hours / Week", "value": "selfstudy"},
]
jobuni_card = white_card(
    html.Div([
        html.Label("Feature vs Job/Uni Status", className="fw-semibold mb-2"),
        dcc.Dropdown(id="y-jobuni", options=jobuni_y_options, value="stress", clearable=False, className="mb-3"),
        dcc.Graph(id="fig-stress-jobuni", config={"displayModeBar": False}, style={"height": "320px"}),
    ]), pad=12
)

continued_col     = dbc.Col(white_card(dcc.Graph(id="fig-continued",     config={"displayModeBar": False}, style={"height": "320px"})), md=12, lg=6)
stress_course_col = dbc.Col(white_card(dcc.Graph(id="fig-stress-course", config={"displayModeBar": False}, style={"height": "320px"})), md=12, lg=6)
focus_line_col    = dbc.Col(white_card(dcc.Graph(id="fig-focus-line",    config={"displayModeBar": False}, style={"height": "320px"})), md=12, lg=6)
exam_line_col     = dbc.Col(white_card(dcc.Graph(id="fig-exam-line",     config={"displayModeBar": False}, style={"height": "320px"})), md=12, lg=6)

layout = dbc.Container(
    [
        html.H2("Insights", className="mt-4"),
        wave_compare_block,
        dbc.Row([continued_col, stress_course_col], class_name="gx-4 gy-4 align-items-start mt-2"),
        dbc.Row([dbc.Col(jobuni_card, md=12, lg=12)], class_name="gx-4 gy-4 align-items-start mt-1"),
        controls,
        dbc.Row([focus_line_col, exam_line_col], class_name="gx-4 gy-4 align-items-start"),
    ],
    fluid=True, className="ps-4 pe-4"
)

# Helpers
def _empty_fig(title, note="No data / column missing"):
    fig = go.Figure()
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.add_annotation(text=note, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(color="#7a8aa5"))
    return fig

def _norm_txt_series(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower().str.strip()
            .str.replace(r"[_\-]+", " ", regex=True)
            .str.replace("/", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True))

def _percent_bins(series, labels=("0–20%","20–40%","40–60%","60–80%","80–100%")):
    v = pd.to_numeric(series, errors="coerce")
    lab_list = list(labels)
    if v.notna().sum() == 0:
        cats = pd.Categorical(values=[np.nan]*len(v), categories=lab_list, ordered=True)
        return pd.Series(cats, index=series.index, name="X"), lab_list
    vmax = float(np.nanmax(v.to_numpy()))
    scale = 100.0 if vmax > 1.5 else 1.0
    edges = np.linspace(0.0, scale, num=len(lab_list) + 1)
    edges[-1] = scale + 1e-12
    cats = pd.cut(v, bins=edges, labels=lab_list, include_lowest=True, right=True, ordered=True)
    return pd.Series(cats, index=series.index, name="X"), lab_list

def _smart_bins(series):
    x, order = _percent_bins(series)
    if pd.isna(x).all():
        v = pd.to_numeric(series, errors="coerce")
        if set(np.unique(v.dropna())) <= {0, 1}:
            labels = ["No","Yes"]
            cats = pd.Categorical(np.where(v == 1, "Yes", np.where(v == 0, "No", np.nan)),
                                  categories=labels, ordered=True)
            return pd.Series(cats, index=series.index, name="X"), labels
        try:
            q = pd.qcut(v, q=[0,.2,.4,.6,.8,1.0], duplicates="drop")
            cats = q.astype(str)
            order = list(cats.dropna().unique())
            cats = pd.Categorical(cats, categories=order, ordered=True)
            return pd.Series(cats, index=series.index, name="X"), order
        except Exception:
            labels = ["All"]; cats = pd.Categorical(["All"]*len(series), categories=labels, ordered=True)
            return pd.Series(cats, index=series.index, name="X"), labels
    return x, order

def _to_percent_numeric(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        vmax = float(np.nanmax(s.to_numpy()))
        return s * 100.0 if vmax <= 1.5 else s
    x = _norm_txt_series(series)
    digs = pd.to_numeric(x.str.extract(r"(\d+(?:\.\d+)?)", expand=False), errors="coerce")
    if digs.notna().any():
        return digs
    map_words = {"none":0,"never":0,"no":0,"low":25,"rarely":25,"some":40,"partial":50,"medium":50,
                 "sometimes":50,"often":70,"mostly":80,"high":80,"all":100,"always":100,"full":100,"every lecture":100}
    return x.map(map_words)

def _map_levels(series: pd.Series, kind: str):
    s = _norm_txt_series(series)
    if kind == "video":
        order = ["never","sometimes","mostly","every_lecture"]
        mapping = {"never":"never","none":"never","no":"never",
                   "rarely":"sometimes","seldom":"sometimes","occasionally":"sometimes","sometimes":"sometimes",
                   "often":"mostly","frequent":"mostly","frequently":"mostly","mostly":"mostly","most":"mostly",
                   "every lecture":"every_lecture","every lectures":"every_lecture","always":"every_lecture","all":"every_lecture"}
    elif kind == "reading":
        order = ["none","partial","mostly","all"]
        mapping = {"none":"none","no":"none","some":"partial","partly":"partial","partially":"partial","partial":"partial",
                   "most":"mostly","mostly":"mostly","all":"all","complete":"all","completed":"all","fully":"all"}
    else:
        raise ValueError("Unknown kind for level mapping")
    mapped = s.map(mapping)
    cats = pd.Categorical(mapped, categories=order, ordered=True)
    return pd.Series(cats, index=series.index, name="X"), order

# Figures
def continued_after_failure_fig(d):
    title = "Students Who Continued After Failure"
    if d.empty or WAVE_COL is None:
        return _empty_fig(title)
    tmp = d.copy()
    failed_mask = pd.to_numeric(tmp.get(AVG_EXAM_COL or "average_exam_score", np.nan), errors="coerce") < 70
    if CONTINUE_COL and CONTINUE_COL in tmp.columns:
        cont = tmp[CONTINUE_COL].astype(str).str.lower().isin(["yes","true","1","continued"])
        tmp = tmp[failed_mask & cont]
    else:
        tmp = tmp[failed_mask]
    if tmp.empty:
        return _empty_fig(title, "No continued-after-failure records")
    counts = (tmp[WAVE_COL].astype(str).value_counts()
              .rename_axis("Wave").reset_index(name="Students").sort_values("Wave"))
    fig = px.bar(counts, x="Wave", y="Students", title=title, text="Students")
    ymax = float(counts["Students"].max())
    fig.update_traces(marker_color=CYAN, textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10,r=10,t=50,b=10), yaxis_title="Number of Students",
                      xaxis_title="Wave", bargap=0.2,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(range=[0, ymax * 1.18])
    return fig

def avg_stress_by_course_fig(d):
    title = "Average Stress by Course"
    if d.empty or COURSE_COL is None or not STRESS_COL or STRESS_COL not in d.columns:
        return _empty_fig(title)
    tmp = d[[COURSE_COL, STRESS_COL]].copy()
    tmp[STRESS_COL] = pd.to_numeric(tmp[STRESS_COL], errors="coerce")
    g = (tmp.groupby(COURSE_COL, observed=True)[STRESS_COL]
           .agg(["mean","count"]).reset_index().sort_values("mean", ascending=False))
    if g.empty:
        return _empty_fig(title)
    g["label"] = g["mean"].round(2).astype(str)
    fig = px.bar(g, x="mean", y=COURSE_COL, orientation="h", title=title, text="label")
    fig.update_traces(marker_color=PURPLE, textposition="outside", cliponaxis=False)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Average Stress Level",
                      yaxis_title="Course", bargap=0.25,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def _add_jobuni_status(tmp: pd.DataFrame) -> pd.DataFrame | None:
    order = ["Job + Uni", "Job Only", "Neither", "Uni Only"]
    if JOBUNI_COL and JOBUNI_COL in tmp.columns:
        raw = (tmp[JOBUNI_COL].astype(str).str.lower()
               .str.replace("&", "+", regex=False)
               .str.replace("/", "+", regex=False))
        mapping = {"job + uni":"Job + Uni","job+uni":"Job + Uni","job and uni":"Job + Uni",
                   "job only":"Job Only","job":"Job Only",
                   "uni only":"Uni Only","uni":"Uni Only",
                   "neither":"Neither","none":"Neither"}
        tmp["status"] = raw.map(mapping).fillna(tmp[JOBUNI_COL].astype(str).str.title())
        tmp.loc[~tmp["status"].isin(order), "status"] = np.nan
    elif JOB_COL or UNI_COL:
        def to_bool(s, true_words):
            x = (s.astype(str).str.lower().str.strip()
                 .str.replace("&", " ", regex=False)
                 .str.replace("/", " ", regex=False)
                 .str.replace("_", " ", regex=False)
                 .str.replace("-", " ", regex=False)
                 .str.replace(r"\s+", " ", regex=True))
            m = False
            for w in true_words:
                m = m | x.str.contains(fr"\b{re.escape(w)}\b", na=False)
            m = m | x.isin(["1","true","yes","y","t"])
            return m
        job_words = ["job","work","working","employed","employment","full time","part time"]
        uni_words = ["uni","university","student","studying","enrolled","enrolment","enrollment"]
        job = to_bool(tmp[JOB_COL], job_words) if JOB_COL and JOB_COL in tmp.columns else False
        uni = to_bool(tmp[UNI_COL], uni_words) if UNI_COL and UNI_COL in tmp.columns else False
        tmp["status"] = np.select([job & uni, job & ~uni, ~job & uni],
                                  ["Job + Uni", "Job Only", "Uni Only"], default="Neither")
    else:
        return None
    tmp["status"] = pd.Categorical(tmp["status"], categories=["Job + Uni","Job Only","Neither","Uni Only"], ordered=True)
    return tmp

def jobuni_metric_fig(d: pd.DataFrame, metric_key: str):
    label_map = {
        "stress":    ("Average Stress Level",    STRESS_COL,   FUCHSIA),
        "focus":     ("Average Focus Level",     FOCUS_COL,    FUCHSIA),
        "selfstudy": ("Self Study Hours / Week", SELF_STD_COL, FUCHSIA),
    }
    title_y, col, color = label_map.get(metric_key, label_map["stress"])
    title = f"{title_y} by Job/Uni Status"
    if d.empty or col is None or col not in d.columns:
        return _empty_fig(title)
    tmp = d.copy()
    tmp = _add_jobuni_status(tmp)
    if tmp is None:
        return _empty_fig(title, "Missing job/uni flags")
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    g = (tmp.groupby("status", observed=True)[col].mean().reset_index(name="avg").dropna())
    if g.empty:
        return _empty_fig(title, "No data after filtering")
    fig = px.line(g, x="status", y="avg", markers=True, title=title)
    fig.update_traces(line=dict(width=3, color=color), marker=dict(size=9, color=color),
                      text=g["avg"].round(2), textposition="top center")
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title="Job/Uni Status", yaxis_title=title_y,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def wave_participation_compare_fig(d: pd.DataFrame):
    title = "Practice Participation Rate by Wave"
    if d.empty or not WAVE_COL:
        return _empty_fig(title, "No data")
    if PARTIC_COL is None or PARTIC_COL not in d.columns:
        return _empty_fig(title, "Participation column not found")
    tmp = d.copy()
    tmp["_pct"] = _to_percent_numeric(tmp[PARTIC_COL])
    g = (tmp.groupby(tmp[WAVE_COL].astype(str), observed=True)["_pct"]
           .mean().reset_index(name="avg_pct")).sort_values(WAVE_COL)
    if g.empty or g["avg_pct"].isna().all():
        return _empty_fig(title, "No numeric participation values")
    fig = px.bar(g, x=WAVE_COL, y="avg_pct", text="avg_pct", title=title)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_color="#3ed4ae")
    fig.update_layout(title_x=0.02, yaxis=dict(title="Participation Rate (%)", range=[0, 100]),
                      xaxis_title="Wave", margin=dict(l=10, r=10, t=40, b=10),
                      bargap=0.25, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig

def focus_line(d, x_key):
    title = "Average Focus Level"
    if d.empty or not FOCUS_COL or FOCUS_COL not in d.columns:
        return _empty_fig(title)
    tmp = d.copy()
    tmp[FOCUS_COL] = pd.to_numeric(tmp[FOCUS_COL], errors="coerce")
    if x_key == "age":
        if AGE_COL is None or AGE_COL not in tmp.columns:
            return _empty_fig(title, "Age column not found")
        bins = [13,16,20,25,30,np.inf]; labels = ["13-15","16-20","21-25","26-30","30+"]
        tmp["X"] = pd.cut(pd.to_numeric(tmp[AGE_COL], errors="coerce"),
                          bins=bins, labels=labels, right=True, include_lowest=True, ordered=True)
    elif x_key == "smbr":
        if SMBR_COL is None or SMBR_COL not in tmp.columns:
            return _empty_fig(title, "Smoking & Breaks column not found")
        tmp["X"] = pd.Categorical(tmp[SMBR_COL], categories=["none","light","frequent"], ordered=True)
    elif x_key == "sleep":
        if SLEEP_COL is None or SLEEP_COL not in tmp.columns:
            return _empty_fig(title, "Sleep hours column not found")
        v = pd.to_numeric(tmp[SLEEP_COL], errors="coerce")
        labels = ["<6h","6–8h","8–10h","10h+"]; edges = [-np.inf,6,8,10,np.inf]
        tmp["X"] = pd.cut(v, bins=edges, labels=labels, include_lowest=True, ordered=True)
    else:
        return _empty_fig(title, "Unknown X selection")
    g = (tmp.groupby("X", observed=True)[FOCUS_COL].mean().reset_index(name="avg").dropna())
    if g.empty:
        return _empty_fig(title, "No data after filtering")
    fig = px.line(g, x="X", y="avg", markers=True, title=title)
    fig.update_traces(line=dict(width=3, color=BLUE), marker=dict(size=9, color=BLUE),
                      text=g["avg"].round(2), textposition="top center")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), xaxis_title="",
                      yaxis_title="Average Focus Level",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

def exam_line(d, x_key):
    title = "Average Exam Score"
    ycol = AVG_EXAM_COL or "average_exam_score"
    if d.empty or ycol not in d.columns:
        return _empty_fig(title)
    tmp = d.copy()
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    if x_key == "attendance":
        if ATT_BIN_COL and ATT_BIN_COL in tmp.columns:
            series = tmp[ATT_BIN_COL].astype(str)
            expected = ["0–20%","20–40%","40–60%","60–80%","80–100%"]
            if set(series.unique()).issubset(set(expected)):
                tmp["X"] = pd.Categorical(series, categories=expected, ordered=True)
            else:
                tmp["X"] = series
        elif ATT_PCT_COL and ATT_PCT_COL in tmp.columns:
            tmp["X"], order = _percent_bins(tmp[ATT_PCT_COL],
                                            labels=("0–20%","20–40%","40–60%","60–80%","80–100%"))
            tmp["X"] = pd.Categorical(tmp["X"], categories=order, ordered=True)
        else:
            return _empty_fig(title, "Attendance column not found")
    elif x_key == "participation":
        if PARTIC_COL is None or PARTIC_COL not in tmp.columns:
            return _empty_fig(title, "Participation column not found")
        tmp["X"], order = _smart_bins(tmp[PARTIC_COL])
        tmp["X"] = pd.Categorical(tmp["X"], categories=order, ordered=True)
    elif x_key == "video":
        if VIDEO_LVL_COL and VIDEO_LVL_COL in tmp.columns:
            tmp["X"], order = _map_levels(tmp[VIDEO_LVL_COL], kind="video")
        elif VIDEO_COL and VIDEO_COL in tmp.columns:
            vnum = pd.to_numeric(tmp[VIDEO_COL], errors="coerce")
            if vnum.notna().sum() > 0:
                tmp["X"], order = _smart_bins(vnum)
            else:
                tmp["X"], order = _map_levels(tmp[VIDEO_COL], kind="video")
        else:
            return _empty_fig(title, "Video engagement column not found")
        tmp["X"] = pd.Categorical(tmp["X"], categories=order, ordered=True)
    elif x_key == "selfstudy":
        if SELF_STD_COL is None or SELF_STD_COL not in tmp.columns:
            return _empty_fig(title, "Self study hours column not found")
        v = pd.to_numeric(tmp[SELF_STD_COL], errors="coerce")
        labels = ["0–2h","2–4h","4–6h","6h+"]; edges = [-np.inf,2,4,6,np.inf]
        tmp["X"] = pd.cut(v, bins=edges, labels=labels, include_lowest=True, ordered=True)
    elif x_key == "reading":
        if READING_LVL_COL and READING_LVL_COL in tmp.columns:
            tmp["X"], order = _map_levels(tmp[READING_LVL_COL], kind="reading")
        elif READING_COL and READING_COL in tmp.columns:
            vnum = pd.to_numeric(tmp[READING_COL], errors="coerce")
            if vnum.notna().sum() > 0:
                tmp["X"], order = _smart_bins(vnum)
            else:
                tmp["X"], order = _map_levels(tmp[READING_COL], kind="reading")
        else:
            return _empty_fig(title, "Reading completion column not found")
        tmp["X"] = pd.Categorical(tmp["X"], categories=order, ordered=True)
    else:
        return _empty_fig(title, "Unknown X selection")
    g = (tmp.groupby("X", observed=True)[ycol].mean().reset_index(name="avg").dropna())
    if g.empty:
        return _empty_fig(title, "No data after filtering")
    fig = px.line(g, x="X", y="avg", markers=True, title=title)
    fig.update_traces(line=dict(width=3, color=BLUE), marker=dict(size=9, color=BLUE),
                      text=g["avg"].round(2), textposition="top center")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), xaxis_title="",
                      yaxis_title="Average Exam Score",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# Callback
@callback(
    Output("fig-wave-compare", "figure"),
    Output("fig-continued",   "figure"),
    Output("fig-stress-course","figure"),
    Output("fig-stress-jobuni","figure"),
    Output("fig-focus-line", "figure"),
    Output("fig-exam-line", "figure"),
    Input("_pages_location", "pathname"),
    Input("filter-wave", "value"),
    Input("filter-status", "value"),
    Input("filter-course", "value"),
    Input("filter-year", "value"),
    Input("y-jobuni", "value"), 
    Input("x-focus", "value"),
    Input("x-exam", "value"),
    Input("data-store", "data"),
    prevent_initial_call=False
)
def update_insights(pathname, wave_sel, status_sel, course_sel, year_sel, y_jobuni, x_focus, x_exam, data_store):
    if pathname != "/insights":
        raise dash.exceptions.PreventUpdate

    if data_store:
        obj = json.loads(data_store["df"])    
        df = pd.DataFrame(obj["data"], columns=obj["columns"])
        COLS = data_store["cols"]
    else:
        df, COLS = DEFAULT_DF, DEFAULT_COLS

    global WAVE_COL, COURSE_COL, AGE_COL, START_COL, END_COL
    global AVG_EXAM_COL, STRESS_COL, FOCUS_COL, SMBR_COL, SLEEP_COL, ATT_BIN_COL, ATT_PCT_COL
    global PARTIC_COL, VIDEO_COL, VIDEO_LVL_COL, READING_COL, READING_LVL_COL, SELF_STD_COL, CONTINUE_COL
    global JOBUNI_COL, JOB_COL, UNI_COL

    WAVE_COL   = COLS.get("WAVE");   COURSE_COL = COLS.get("COURSE"); AGE_COL = COLS.get("AGE")
    START_COL  = COLS.get("START");  END_COL    = COLS.get("END")
    AVG_EXAM_COL = _pick_col(df, ["average_exam_score","avg_exam","exam_avg","exam_score_avg"])
    STRESS_COL   = _pick_col(df, ["stress_level","stress","avg_stress","stressscore"])
    FOCUS_COL    = _pick_col(df, ["focus_level","focus"])
    SMBR_COL     = _pick_col(df, ["smoking_breaks_level","smoking_breaks","breaks_level"])
    SLEEP_COL    = _pick_col(df, ["sleep_hours","sleep","sleep_hours_per_night","sleepdurationhours","sleep_duration"])
    ATT_BIN_COL  = _pick_col(df, ["attendance_bin","lecture_attendance_bin"])
    ATT_PCT_COL  = _pick_col(df, ["lecture_attendance","attendance_percent","attendance_pct","attendance_percentage"])
    PARTIC_COL   = _pick_col(df, ["participation_rate","class_participation","participationpct",
                                  "participation","participation_percent","participation_percentage"])
    VIDEO_COL    = _pick_col(df, ["video_lecture_engagement","video_engagement","video_engagement_rate",
                                  "video_engagement_percent","video_engagement_percentage","video_lecture_completion",
                                  "video_lectures_completion","video_watched_percent","video_watch_rate","video_watched_pct"])
    VIDEO_LVL_COL   = _pick_col(df, ["video_lecture_engagement_level","video_level","video_engagement_level",
                                     "video_lectures_engagement_level"])
    READING_COL     = _pick_col(df, ["assigned_reading_completion","reading_completion","reading_completion_rate",
                                     "reading_completion_percent","reading_completion_percentage","assigned_reading_completion_rate",
                                     "assigned_reading_completion_pct"])
    READING_LVL_COL = _pick_col(df, ["assigned_readings_completion_level","assigned_reading_completion_level",
                                     "reading_completion_level","reading_level"])
    SELF_STD_COL    = _pick_col(df, ["self_study_hours","selfstudy_hours","study_hours"])
    CONTINUE_COL    = _pick_col(df, ["continued_after_failure","continued","continue_status"])
    JOBUNI_COL      = _pick_col(df, ["job_uni_status","jobunistatus","job_uni","job and uni",
                                     "employment_study_status","study_job_status","jobuni"])
    JOB_COL         = _pick_col(df, ["job","job_status","has_job","employment_status","employed",
                                     "work_status","working","jobflag"])
    UNI_COL         = _pick_col(df, ["uni","university","is_student","student_status","enrolled",
                                     "enrollment_status","study_status","studying","uni_enrolled"])

    # Filtered data
    d = apply_filters(df, COLS,
                      wave_sel or "All",
                      status_sel or "All",
                      course_sel or "All",
                      year_sel or "All",
                      TODAY)

    return (
        wave_participation_compare_fig(d),
        continued_after_failure_fig(d),
        avg_stress_by_course_fig(d),
        jobuni_metric_fig(d, y_jobuni or "stress"),  
        focus_line(d, x_focus or "age"),
        exam_line(d,  x_exam  or "attendance"),
    )
