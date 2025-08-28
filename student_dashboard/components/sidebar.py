import os, io, base64, json
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input, State, no_update

from pages.overview_parts import (
    build_options,
    prepare_dataframe,         
    store_payload,            
)

def nav_item(name: str, path: str, icon: str):
    return dbc.NavLink(
        html.Span([html.I(className=f"bi {icon} me-2"), name]),
        href=path,
        active="exact",
        class_name="side-link"
    )

def build_sidebar(extra_panel=None):
    pages = {p["path"]: p for p in dash.page_registry.values()}
    order = ["/", "/insights"]

    icon_for = {"/": "bi-house-fill", "/insights": "bi-graph-up-arrow"}
    items = []
    for pth in order:
        if pth in pages:
            page = pages[pth]
            items.append(nav_item(page["name"], page["path"], icon_for.get(pth, "bi-circle")))
    for page in sorted([p for p in pages.values() if p["path"] not in order],
                       key=lambda p: (p.get("order", 999), p["name"])):
        items.append(nav_item(page["name"], page["path"], "bi-circle"))

    inner_children = [
        html.Div([
            html.Div(className="brand-avatar", children=html.I(className="bi bi-stars")),
            html.Div(className="brand-text", children=[
                html.Div("Picsart", className="brand-title"),
                html.Div("Student Analytics", className="brand-subtitle")
            ])
        ], className="brand-row"),

        html.Div(dbc.Nav(items, vertical=True, pills=True, class_name="sidebar-nav"),
                 className="sidebar-subcard"),
    ]

    if extra_panel is not None:
        inner_children.append(html.Div(extra_panel, className="sidebar-subcard mt-3"))

    inner_children.append(
    html.Div(
        [
            dcc.Upload(
                id="upload-data",
                children=html.Span([html.I(className="bi bi-upload me-2"), "Upload CSV"]),
                multiple=False,
                accept=".csv",
                className="upload-pill",
                style={"border": "none", "width": "100%", "display": "block"},
            ),
            html.Div(id="upload-status", className="text-muted small mt-2"),
        ],
        className="sidebar-subcard upload-subcard mt-3",
    )
)

    return html.Aside(dbc.Card(dbc.CardBody(inner_children), class_name="sidebar-card"),
                      className="sidebar")

@callback(
    Output("filter-wave",   "value", allow_duplicate=True),
    Output("filter-status", "value", allow_duplicate=True),
    Output("filter-course", "value", allow_duplicate=True),
    Output("filter-year",   "value", allow_duplicate=True),
    Input("reset-filters",  "n_clicks"),
    prevent_initial_call=True
)
def _reset_filters(n):
    return ["All"], "All", ["All"], "All"

@callback(
    Output("data-store", "data"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)

def _handle_upload(contents, filename):
    if not contents or not filename:
        return no_update, no_update

    try:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df_raw = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception as e:
        return no_update, f"Upload failed: {e}"

    df, COLS = prepare_dataframe(df_raw)
    payload = store_payload(df, COLS)

    import json, os
    os.makedirs("data", exist_ok=True)
    df.to_csv(os.path.join("data", "uploaded_latest.csv"), index=False)

    with open(os.path.join("data", "last_payload.json"), "w", encoding="utf-8") as f:
        json.dump({"df": payload["df"], "cols": payload["cols"]}, f)

    msg = f"Loaded {filename}  —  {len(df):,} rows, {df.shape[1]} cols"
    return payload, msg

@callback(
    Output("filter-wave",   "options", allow_duplicate=True),
    Output("filter-course", "options", allow_duplicate=True),
    Output("filter-year",   "options", allow_duplicate=True),
    Output("filter-wave",   "value",   allow_duplicate=True),
    Output("filter-course", "value",   allow_duplicate=True),
    Output("filter-year",   "value",   allow_duplicate=True),
    Input("data-store",     "data"),
    prevent_initial_call=True
)

def _refresh_filter_options(data_store):
    import json
    import pandas as pd

    if not data_store:
        raise dash.exceptions.PreventUpdate

    obj = json.loads(data_store["df"])
    df  = pd.DataFrame(obj["data"], columns=obj["columns"])
    COLS = data_store["cols"]

    start_col = COLS.get("START")
    if start_col and start_col in df.columns:
        df[start_col] = pd.to_datetime(df[start_col], errors="coerce")

    wave_opts, course_opts, year_opts = build_options(df, COLS)

    def normalize(opts):
        out = []
        for o in opts:
            out.append(o if isinstance(o, dict) else {"label": str(o), "value": str(o)})
        return out

    return (
        normalize(wave_opts),
        normalize(course_opts),
        normalize(year_opts),
        ["All"], ["All"], "All"
    )