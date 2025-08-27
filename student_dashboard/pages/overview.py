import numpy as np
import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import io, json

from .overview_parts import (
    load_df, build_options, apply_filters, make_filters,
    kpi_card, donut_figure, gender_figure, age_figure, passfail_figure
)

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

dash.register_page(__name__, path="/", name="Overview", order=0)

DEFAULT_DF, DEFAULT_COLS = load_df()
TODAY = pd.Timestamp.today().normalize()

wave_options, course_options, year_options = build_options(DEFAULT_DF, DEFAULT_COLS)
filters = make_filters(wave_options, course_options, year_options)

# KPI row
kpi_row = dbc.Row(
    [
        dbc.Col(kpi_card("bi-mortarboard-fill", "Total Students Enrolled", "kpi-total"), width="auto"),
        dbc.Col(kpi_card("bi-journal-bookmark-fill", "Top Course by Enrollments", "kpi-topcourse"), width="auto"),
        dbc.Col(kpi_card("bi-calendar-event", "Year with Most Enrollments", "kpi-peak-year"), width="auto"),
    ],
    class_name="g-2",
)

# Charts 
gender_chart = dbc.Col(white_card(dcc.Graph(id="fig-gender",
                                            config={"displayModeBar": False},
                                            style={"height": "250px"})), md=12, lg=6)

age_chart = dbc.Col(white_card(dcc.Graph(id="fig-age",
                                         config={"displayModeBar": False},
                                         style={"height": "250px"})), md=12, lg=6)

avg_panel = dbc.Col(white_card(html.Div(id="avg-exams-col")), md=12, lg=12)

pass_fail_chart = dbc.Col(white_card(dcc.Graph(id="fig-passfail",
                                               config={"displayModeBar": False},
                                               style={"height": "300px"})), md=12)

background_chart = dbc.Col(
    html.Div(
        dcc.Graph(id="fig-background", config={"displayModeBar": False}, style={"height": "300px"}),
        style={
            "backgroundColor": "white",
            "border": "1px solid #e0e6f1",
            "borderRadius": "12px",
            "padding": "15px",
            "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.05)",
        },
    ),
    md=12, lg=6
)

enroll_year_chart = dbc.Col(
    html.Div(
        dcc.Graph(id="fig-enroll-year", config={"displayModeBar": False}, style={"height": "300px"}),
        style={
            "backgroundColor": "white",
            "border": "1px solid #e0e6f1",
            "borderRadius": "12px",
            "padding": "15px",
            "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.05)",
        },
    ),
    md=12, lg=6
)

charts_top_row = dbc.Row([gender_chart, age_chart], class_name="gx-4 gy-4 align-items-start mt-2")
charts_mid_row = dbc.Row([background_chart, enroll_year_chart], class_name="gx-4 gy-4 align-items-start mt-2")
charts_bottom_row = dbc.Row([avg_panel], class_name="align-items-start mt-3")
pass_fail_row = dbc.Row([pass_fail_chart], class_name="align-items-start mt-3")

layout = dbc.Container(
    [
        html.H2("Overview", className="mt-4"),
        kpi_row,
        charts_top_row,       # Gender & Age
        charts_bottom_row,    # Average exam donuts
        pass_fail_row,        # Pass/Fail
        charts_mid_row,       # Background & Enrollment-by-Year
    ],
    fluid=True, className="ps-4 pe-4"
)

@callback(
    Output("kpi-total","children"),
    Output("kpi-topcourse","children"),
    Output("kpi-peak-year","children"),
    Output("fig-gender","figure"),
    Output("fig-age","figure"),
    Output("fig-background","figure"),
    Output("fig-enroll-year","figure"),
    Output("fig-passfail","figure"),
    Output("avg-exams-col","children"),
    Input("filter-wave","value"),
    Input("filter-status","value"),
    Input("filter-course","value"),
    Input("filter-year","value"),
    Input("data-store","data"),
    prevent_initial_call=False
)
def update_overview(pathname, wave_sel, status_sel, course_sel, year_sel, data_store):
    if pathname != "/":
        raise dash.exceptions.PreventUpdate

    if data_store:
        obj = json.loads(data_store["df"])        
        df  = pd.DataFrame(obj["data"], columns=obj["columns"])
        COLS = data_store["cols"]
    else:
        df, COLS = DEFAULT_DF, DEFAULT_COLS

    WAVE_COL   = COLS.get("WAVE")
    COURSE_COL = COLS.get("COURSE")
    GENDER_COL = COLS.get("GENDER")
    AGE_COL    = COLS.get("AGE")
    START_COL  = COLS.get("START")
    END_COL    = COLS.get("END")
    BG_COL     = COLS.get("BG")

    d = apply_filters(df, COLS,
                      wave_sel or "All",
                      status_sel or "All",
                      course_sel or "All",
                      year_sel or "All",
                      TODAY)
    
    if START_COL and START_COL in d.columns:
        d[START_COL] = pd.to_datetime(d[START_COL], errors="coerce")
    if END_COL and END_COL in d.columns:
        d[END_COL] = pd.to_datetime(d[END_COL], errors="coerce")


    # KPIs
    total_students = len(d)

    # Peak Enrollment Year
    if START_COL and START_COL in d.columns:
        year_series = d[START_COL].dropna().dt.year.astype(int)
    elif "_start_year" in d.columns:
        year_series = d["_start_year"].dropna().astype(int)
    else:
        year_series = pd.Series([], dtype=int)

    if not year_series.empty:
        vc_year = year_series.value_counts()
        peak_year_value = str(int(vc_year.idxmax()))
    else:
        peak_year_value = "N/A"

    # Top Course
    top_course = "N/A"
    if COURSE_COL and not d.empty and COURSE_COL in d.columns:
        vc = d[COURSE_COL].value_counts()
        if not vc.empty:
            top_course = str(vc.idxmax())

    # Figures 
    fig_gender   = gender_figure(d, GENDER_COL)
    fig_age      = age_figure(d, AGE_COL)
    fig_passfail = passfail_figure(d, COURSE_COL)

    # Background donut
    if BG_COL and not d.empty and BG_COL in d.columns:
        bg = (d[BG_COL].astype(str).str.strip().str.lower()
              .replace({'^beg.*':'beginner', '^inter.*':'intermediate', '^adv.*':'advanced'}, regex=True))
        bg_counts = (bg.value_counts().rename_axis("level").reset_index(name="count"))
        bg_counts["label"] = bg_counts["level"].str.title()
        color_map = {"beginner": "#3A72EF", "intermediate": "#D500C8", "advanced": "#8F57EF"}

        fig_background = px.pie(
            bg_counts, values="count", names="label", hole=0.65, color="level",
            color_discrete_map=color_map, title="Students by Background Level"
        )
        fig_background.update_traces(
            pull=[0.02]*len(bg_counts),
            marker=dict(line=dict(color="#FFFFFF", width=4)),
            sort=False, textposition="outside", textinfo="label+percent",
        )
        fig_background.add_annotation(
            text=f"{int(bg_counts['count'].sum())}<br>Students",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#24314A")
        )
        fig_background.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig_background = px.pie(title="Students by Background Level")
        fig_background.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Enrollment by Year
    if START_COL and START_COL in d.columns:
        year_series = d[START_COL].dropna().dt.year.astype(int)
    elif "_start_year" in d.columns:
        year_series = d["_start_year"].dropna().astype(int)
    else:
        year_series = pd.Series([], dtype=int)

    if not year_series.empty:
        year_counts = (year_series.value_counts()
                       .rename_axis("Year")
                       .reset_index(name="Enrollment")
                       .sort_values("Year"))
        fig_enroll_year = px.line(year_counts, x="Year", y="Enrollment", markers=True,
                                  title="Students Enrollment by Year")
        fig_enroll_year.update_traces(line=dict(width=3), marker=dict(size=8),
                                      text=year_counts["Enrollment"], textposition="top center")
        fig_enroll_year.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(dtick=1, title="Year"),
            yaxis_title="Enrollment",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
    else:
        fig_enroll_year = px.line(title="Students Enrollment by Year")
        fig_enroll_year.update_layout(margin=dict(l=10, r=10, t=40, b=10),
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

    # Average-score donuts
    def avg_for(name: str):
        if "average_exam_score" not in d.columns or d.empty or COURSE_COL is None:
            return None
        block = d[d[COURSE_COL].astype(str).str.casefold() == name.casefold()]
        vals = pd.to_numeric(block["average_exam_score"], errors="coerce")
        return float(vals.mean().round(2)) if vals.notna().any() else None

    figs = [
        ("DevHack", donut_figure("Average Exam Score: DevHack", avg_for("DevHack"), "#3a72ef")),
        ("AI",      donut_figure("Average Exam Score: AI",       avg_for("AI"),      "#8f57ef")),
        ("C++",     donut_figure("Average Exam Score: C++",      avg_for("C++"),     "#FF45FC")),
        ("Web",     donut_figure("Average Exam Score: Web",      avg_for("Web"),     "#36e5ff")),
    ]

    sel_list = course_sel if isinstance(course_sel, (list, tuple, set)) else [course_sel]
    sel_list = [str(x).strip() for x in (sel_list or []) if str(x).strip()]
    if sel_list and not any(s.lower() == "all" for s in sel_list):
        sel_set = {s.casefold() for s in sel_list}
        figs = [t for t in figs if t[0].casefold() in sel_set]

    def cell(fig):
        if not getattr(fig, "data", None):
            return None
        return dbc.Col(
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "190px", "width": "100%"}),
            xs=12, sm=6, md=6, lg=3, class_name="d-flex justify-content-center px-3"
        )

    avg_exams_children = dbc.Row([c for _, f in figs if (c := cell(f))],
                                 class_name="gx-5 gy-3 justify-content-center")

    return (f"{total_students}", top_course, peak_year_value,
            fig_gender, fig_age, fig_background, fig_enroll_year, fig_passfail, avg_exams_children)
