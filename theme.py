"""MIRAGE design tokens, CSS and Plotly template."""
import plotly.graph_objects as go
import plotly.io as pio

C = dict(canvas="#F5F3EE", paper="#FBFAF7", ink="#20211E", ink2="#65665F", muted="#888980", line="#DCD9D0", subtle="#ECE9E1",
         olive="#777D68", deep="#555B49", green="#49745A", amber="#A8793E", red="#A6544D", cool="#777D80")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; color:{C['ink']}; }}
.stApp {{ background:{C['canvas']}; }}
header[data-testid="stHeader"] {{ display:none; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ max-width:1180px; padding:1.4rem 1.5rem 4rem 1.5rem; }}
h1,h2,h3 {{ font-family:'Instrument Serif', Georgia, serif; font-weight:400; letter-spacing:-0.01em; color:{C['ink']}; }}
.wordmark {{ font-family:'Instrument Serif', Georgia, serif; font-size:1.7rem; line-height:1; }}
.wordmark span {{ color:{C['olive']}; }}
.status {{ font-size:.78rem; color:{C['ink2']}; text-align:right; padding-top:.5rem; }}
.dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:{C['green']}; margin-right:6px; }}
.dot.err {{ background:{C['red']}; }}
.eyebrow {{ font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:{C['muted']}; margin:1.6rem 0 .4rem; font-weight:500; }}
.hero {{ font-family:'Instrument Serif', Georgia, serif; font-size:3rem; line-height:1.05; margin:.2rem 0 .6rem; }}
.hero em {{ color:{C['deep']}; }}
.lede {{ color:{C['ink2']}; max-width:640px; font-size:1rem; line-height:1.55; }}
.card {{ background:{C['paper']}; border:1px solid {C['line']}; border-radius:6px; padding:1.1rem 1.3rem; margin-bottom:.9rem; }}
.kpi-l {{ font-size:.72rem; color:{C['muted']}; text-transform:uppercase; letter-spacing:.08em; }}
.kpi-v {{ font-family:'Instrument Serif', Georgia, serif; font-size:2rem; line-height:1.15; }}
.kpi-s {{ font-size:.78rem; color:{C['ink2']}; }}
.tag {{ display:inline-block; font-size:.68rem; letter-spacing:.06em; text-transform:uppercase; padding:1px 7px; border:1px solid {C['line']}; border-radius:3px; color:{C['ink2']}; margin-right:6px; background:{C['subtle']}; }}
.tag.obs {{ color:{C['green']}; border-color:{C['green']}; background:transparent; }}
.tag.int {{ color:{C['amber']}; border-color:{C['amber']}; background:transparent; }}
.tag.lim {{ color:{C['cool']}; border-color:{C['cool']}; background:transparent; }}
.tag.flag {{ color:{C['red']}; border-color:{C['red']}; background:transparent; }}
.col-h {{ font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:{C['muted']}; margin-bottom:.4rem; }}
.big {{ font-family:'Instrument Serif', Georgia, serif; font-size:1.55rem; line-height:1.25; }}
.small {{ font-size:.82rem; color:{C['ink2']}; }}
.crumb {{ font-size:.8rem; color:{C['muted']}; margin-bottom:.5rem; }}
.sec {{ font-family:'Instrument Serif', Georgia, serif; font-size:1.5rem; margin:1.4rem 0 .3rem; }}
.stButton>button, .stDownloadButton>button {{ border-radius:4px; border:1px solid {C['line']}; background:{C['paper']}; color:{C['ink']}; font-weight:500; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{C['deep']}; color:{C['deep']}; }}
.stButton>button[kind="primary"] {{ background:{C['deep']}; color:#fff; border-color:{C['deep']}; }}
div[role="radiogroup"] label {{ font-size:.9rem; }}
hr {{ border-color:{C['line']}; }}
@media (max-width: 640px) {{ .hero {{ font-size:2.1rem; }} .block-container {{ padding:1rem 16px 3rem; }} }}
</style>
"""

def plotly_template():
    t = go.layout.Template()
    t.layout = go.Layout(font=dict(family="Inter, sans-serif", color=C["ink2"], size=12), paper_bgcolor=C["paper"], plot_bgcolor=C["paper"],
                         colorway=[C["deep"], C["amber"], C["cool"], C["red"], C["olive"]], margin=dict(l=10, r=10, t=30, b=10),
                         xaxis=dict(gridcolor=C["line"], linecolor=C["line"], zeroline=False), yaxis=dict(gridcolor=C["line"], linecolor=C["line"], zeroline=False),
                         legend=dict(orientation="h", y=-0.2))
    pio.templates["mirage"] = t
    pio.templates.default = "mirage"
