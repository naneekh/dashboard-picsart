import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.MORPH, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True
)

server = app.server

from pages.overview_parts import default_store_payload
DATA_STORE_INIT = default_store_payload() 

import pages.overview
import pages.overview_parts
import pages.insights
from pages.overview import filters as overview_filters
from components.sidebar import build_sidebar

from pages.overview_parts import default_store_payload  # 👈 uses your loader

sidebar = build_sidebar(extra_panel=overview_filters)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            src="/assets/picsart_logo.png",
                            height="64px",
                            className="app-logo"
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.NavbarBrand(
                            "Picsart Academy — Student Analytics",
                            class_name="app-title"
                        ),
                        width="auto",
                    ),
                ],
                align="center",
                class_name="g-2", 
            )
        ],
        fluid=True,
    ),
    class_name="app-header"
)

header = dbc.Navbar(
    dbc.Container(
        [
            html.Img(
                src="/assets/picsart_logo.png", 
                height="50px",
                className="me-3"
            ),
            dbc.NavbarBrand(
                "Picsart Academy — Student Analytics",
                class_name="app-title"
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
    class_name="app-header"
)

app.layout = dbc.Container(fluid=True, children=[
    dcc.Store(id="store", data=default_store_payload()),

    navbar,
    dbc.Row(className="g-2", children=[
        dbc.Col(sidebar, xs=12, md=4, lg=3, xxl=3, className="sidebar-col"),
        dbc.Col(dash.page_container, xs=12, md=8, lg=9, xxl=9),
    ])
])

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)