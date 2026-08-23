"""Shared visual language for the PAIP NRW dashboard.

Light and dark are both *selected*, not flipped. The dark column is the same
eight hues re-stepped for the dark surface, each validated against that surface
rather than derived by inverting the light values.

Palette values follow a validated categorical order (adjacent-pair CVD deltaE >= 8,
normal-vision deltaE >= 15 in both modes). Slots are assigned in fixed order and
never cycled; a 9th series folds into "Other" rather than generating a new hue.
Scatter, bubble and cluster forms are capped at the first three slots, which are
the ones that clear the all-pairs floors.
"""

import plotly.graph_objects as go
import plotly.io as pio

# ---- Categorical slots, both modes ---------------------------------------
SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"]

# ---- Sequential ramp (single hue, light -> dark) -------------------------
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]
# On the dark surface the ramp runs the other way: the step nearest the surface
# must still separate from it, so the light end leads.
SEQ_BLUE_DARK = list(reversed(SEQ_BLUE[:-2]))

# ---- Status (reserved; always paired with an icon or label) --------------
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

# ---- Chrome and ink, per mode -------------------------------------------
# Surfaces carry the only "water" the palette itself contains: a faint cool
# tint rather than neutral grey. Both were re-validated with the categorical
# set against these exact surfaces — contrast and CVD results are meaningless
# against a surface the chart does not actually render on.
PALETTES = {
    "light": dict(
        series=SERIES_LIGHT, seq=SEQ_BLUE,
        surface="#fbfdfe", page="#eef4f9",
        ink="#0b1a24", ink2="#46606f", muted="#6d8695",
        grid="#e2ecf3", baseline="#c3d3dd",
        success_text="#006300", neutral="#c3ced6",
        border="rgba(11,26,36,0.12)", hover_bg="#ffffff",
        table_head="#e7eff7",
        tile_wash="rgba(42,120,214,0.10)",
    ),
    "dark": dict(
        series=SERIES_DARK, seq=SEQ_BLUE_DARK,
        surface="#131c24", page="#0a1016",
        ink="#ffffff", ink2="#c2d2dc", muted="#86a0ad",
        grid="#22303b", baseline="#33454f",
        success_text="#0ca30c", neutral="#4a5b66",
        border="rgba(255,255,255,0.12)", hover_bg="#1c2831",
        table_head="#1b2733",
        tile_wash="rgba(57,135,229,0.16)",
    ),
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

NATIONAL_NRW_PCT = 35.0
POLICY_TARGET_PCT = 25.0


class Theme:
    """Resolved palette for one mode. Charts are written against roles, so the
    whole dashboard re-colours by swapping this object."""

    def __init__(self, mode: str = "light"):
        self.mode = mode if mode in PALETTES else "light"
        p = PALETTES[self.mode]
        self.SERIES = p["series"]
        (self.BLUE, self.ORANGE, self.AQUA, self.YELLOW,
         self.MAGENTA, self.GREEN, self.VIOLET, self.RED) = p["series"]
        self.SEQ = p["seq"]
        self.SURFACE = p["surface"]
        self.PAGE = p["page"]
        self.INK = p["ink"]
        self.INK_2 = p["ink2"]
        self.MUTED = p["muted"]
        self.GRID = p["grid"]
        self.BASELINE = p["baseline"]
        self.SUCCESS_TEXT = p["success_text"]
        self.NEUTRAL = p["neutral"]
        self.BORDER = p["border"]
        self.HOVER_BG = p["hover_bg"]
        self.TILE_WASH = p["tile_wash"]
        self.TABLE_HEAD = p["table_head"]
        self.GOOD, self.WARNING = GOOD, WARNING
        self.SERIOUS, self.CRITICAL = SERIOUS, CRITICAL
        self.POLICY_TARGET_PCT = POLICY_TARGET_PCT
        self.NATIONAL_NRW_PCT = NATIONAL_NRW_PCT
        self.FONT = FONT
        self.template = f"paip_{self.mode}"
        self._install()

    def _install(self):
        tpl = go.layout.Template()
        tpl.layout = go.Layout(
            font=dict(family=FONT, size=12.5, color=self.INK_2),
            title=dict(font=dict(size=16, color=self.INK), x=0,
                       xanchor="left", y=0.97, yanchor="top"),
            paper_bgcolor=self.SURFACE,
            plot_bgcolor=self.SURFACE,
            colorway=self.SERIES,
            margin=dict(l=10, r=10, t=84, b=10),
            hoverlabel=dict(bgcolor=self.HOVER_BG, bordercolor=self.BASELINE,
                            font=dict(family=FONT, size=12, color=self.INK)),
            xaxis=dict(gridcolor=self.GRID, gridwidth=1, zeroline=False,
                       linecolor=self.BASELINE, linewidth=1, ticks="",
                       tickfont=dict(color=self.MUTED, size=11.5),
                       # Long plant names on horizontal bar charts were being
                       # clipped by the 10px margin; automargin grows it instead.
                       automargin=True,
                       title=dict(font=dict(color=self.INK_2, size=12))),
            yaxis=dict(gridcolor=self.GRID, gridwidth=1, zeroline=False,
                       showline=False, ticks="",
                       tickfont=dict(color=self.MUTED, size=11.5),
                       # Long plant names on horizontal bar charts were being
                       # clipped by the 10px margin; automargin grows it instead.
                       automargin=True,
                       title=dict(font=dict(color=self.INK_2, size=12))),
            legend=dict(orientation="h", yanchor="bottom", y=1.015,
                        xanchor="left", x=0,
                        font=dict(size=12, color=self.INK_2),
                        bgcolor="rgba(0,0,0,0)", traceorder="normal"),
        )
        pio.templates[self.template] = tpl
        pio.templates.default = self.template

    # -- HTML helpers ------------------------------------------------------
    def tile(self, label, value, unit="", sub=""):
        u = f'<span class="tile-unit"> {unit}</span>' if unit else ""
        s = f'<div class="tile-sub">{sub}</div>' if sub else ""
        return (f'<div class="tile"><div class="tile-label">{label}</div>'
                f'<div class="tile-value">{value}{u}</div>{s}</div>')

    def callout(self, text, kind=""):
        cls = {"warn": " callout-warn", "crit": " callout-crit",
               "good": " callout-good"}.get(kind, "")
        return f'<div class="callout{cls}">{text}</div>'

    @property
    def css(self):
        p = self
        return f"""
<style>
  .stApp {{ background: {p.PAGE}; }}
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  /* Streamlit's floating header overlays the first ~55px and was clipping the
     brand row, so its height is reclaimed for content. It is FLATTENED, not
     removed: the control that reopens a collapsed sidebar lives inside it, and
     `display:none` on the header zeroes that button — CSS cannot un-hide a
     descendant of a hidden ancestor, so the sidebar becomes unreopenable and
     the only way back is a page reload.

     So: zero height, transparent, click-through — then the expand button alone
     is given pointer events back and pinned as a fixed control. */
  /* !important is required: Streamlit's emotion styles are injected after this
     block and match with equal specificity, so they win on order alone. */
  [data-testid="stHeader"], .stAppHeader {{
    background: transparent !important;
    height: 0 !important; min-height: 0 !important;
    pointer-events: none !important; }}
  /* Everything the flattened header still contains, hidden by name. The
     deploy button is stAppDeployButton in this build, not stDeployButton;
     stBaseButton-header covers the same control if that id shifts again. */
  [data-testid="stToolbarActions"], [data-testid="stMainMenu"],
  [data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
  [data-testid="stBaseButton-header"], [data-testid="stDeployButton"],
  [data-testid="stDecoration"] {{ display: none !important; }}

  [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapsedControl"] {{
    pointer-events: auto !important;
    position: fixed !important; top: 8px !important; left: 8px !important;
    z-index: 1001 !important;
    background: {p.SURFACE} !important;
    border: 1px solid {p.BORDER} !important;
    border-radius: 8px !important; }}

  /* With the sidebar closed the main pane starts at x=0, and that fixed button
     would sit on top of the first KPI tile. Nudge the content clear of it —
     only in the collapsed state. */
  .stApp:has(section[data-testid="stSidebar"][aria-expanded="false"])
    .block-container {{ padding-left: 54px !important; }}
  .block-container, .stMainBlockContainer {{
    padding-top: 1.1rem !important; max-width: 1840px !important; }}

  h1, h2, h3, h4, h5, h6 {{ color: {p.INK}; letter-spacing: -0.01em; }}
  .stApp, .stMarkdown, p, span, label, li {{ color: {p.INK_2}; }}

  .tile {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER};
    border-radius: 10px; padding: 16px 18px 14px 18px; height: 100%;
  }}
  .tile-label {{
    font-size: 11.5px; font-weight: 600; letter-spacing: 0.04em;
    text-transform: uppercase; color: {p.MUTED}; margin-bottom: 6px;
  }}
  .tile-value {{ font-size: 30px; font-weight: 650; color: {p.INK}; line-height: 1.1; }}
  .tile-unit {{ font-size: 15px; font-weight: 500; color: {p.INK_2}; }}
  .tile-sub {{ font-size: 12px; color: {p.INK_2}; margin-top: 6px; line-height: 1.45; }}
  .tile-delta-good {{ color: {p.SUCCESS_TEXT}; font-weight: 600; }}
  .tile-delta-bad {{ color: {p.CRITICAL}; font-weight: 600; }}

  .callout {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER};
    border-left: 3px solid {p.BLUE}; border-radius: 8px;
    padding: 14px 18px; margin: 6px 0 18px 0;
    font-size: 13.5px; color: {p.INK_2}; line-height: 1.6;
  }}
  .callout-warn {{ border-left-color: {WARNING}; }}
  .callout-crit {{ border-left-color: {CRITICAL}; }}
  .callout-good {{ border-left-color: {GOOD}; }}
  .callout b {{ color: {p.INK}; }}

  .caption {{ font-size: 12px; color: {p.MUTED}; line-height: 1.55;
              margin: -6px 0 16px 0; }}
  .caption b {{ color: {p.INK_2}; }}

  .stTabs [role="tablist"] {{ gap: 2px; border-bottom: 1px solid {p.GRID}; }}
  .stTabs [data-testid="stTab"] {{ height: 42px; padding: 0 16px;
                                   background: transparent; }}
  .stTabs [data-testid="stTab"] p {{ font-size: 13.5px; font-weight: 500;
                                     color: {p.MUTED}; margin: 0; }}
  .stTabs [data-testid="stTab"][aria-selected="true"] p {{
    color: {p.INK}; font-weight: 600; }}

  section[data-testid="stSidebar"] {{ background: {p.SURFACE};
    border-right: 1px solid {p.GRID}; }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}

  /* Top strip ---------------------------------------------------------
     Filters live in the sidebar now, so the top strip carries only identity
     and the KPI rail. It must not scroll on a 1080p screen, so both are
     compressed into fixed-height strips. */
  .brandbar {{ display:flex; flex-direction:column; justify-content:center;
               height:38px; }}
  .brand {{ font-size:15px; font-weight:650; color:{p.INK};
            letter-spacing:-0.01em; line-height:1.15; }}
  .brandsub {{ font-size:11px; color:{p.MUTED}; line-height:1.3; }}

  .kpistrip {{ display:flex; gap:8px; margin:2px 0 10px 0; }}
  .kpi {{ flex:1; background:{p.SURFACE}; border:1px solid {p.BORDER};
          border-radius:8px; padding:8px 12px; min-width:0; }}
  .kpi-title {{ flex:1.6; border-left:3px solid {p.BLUE}; }}
  .kpi-l {{ font-size:10px; font-weight:600; letter-spacing:0.05em;
            text-transform:uppercase; color:{p.MUTED}; }}
  .kpi-v {{ font-size:20px; font-weight:650; color:{p.INK}; line-height:1.25;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .kpi-s {{ font-size:10.5px; color:{p.INK_2}; line-height:1.3;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .kpi-good {{ color:{p.SUCCESS_TEXT}; font-weight:600; }}
  .kpi-bad {{ color:{CRITICAL}; font-weight:600; }}

  /* Nested (sub) tabs sit tighter than the main row. */
  .stTabs .stTabs [data-testid="stTab"] {{ height:34px; padding:0 12px; }}
  .stTabs .stTabs [data-testid="stTab"] p {{ font-size:12.5px; }}

  /* Top padding must clear the brand row; the default was cropping it. */
  .block-container {{ padding-bottom: 0.6rem !important; }}

  div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
  hr {{ border-color: {p.GRID}; margin: 1.4rem 0; }}

  /* Accent override ---------------------------------------------------
     config.toml deliberately sets no [theme], because setting ANY theme key
     makes Streamlit resolve a concrete base and forces its widget chrome and
     built-in Plotly theme to light even when the OS is dark. The cost is that
     Streamlit keeps its default red accent (#ff4b4b), which collides with the
     status palette's "critical" and reads as an alert on every chip and
     slider. These rules re-point the accent at the palette blue using stable
     data-testid / react-aria hooks rather than build-specific emotion hashes.

     The :not([data-testid]) guards matter: without them these rules also paint
     the label's stMarkdownContainer and the slider's tick bar, which turns the
     option TEXT into a blue block instead of tinting the control. */
  [data-testid="stRadioOption"] > div > div > div:not([data-testid]) {{
    background-color: {p.BLUE} !important; }}
  [data-testid="stMultiSelectTagsContainer"] span[role="group"] > span {{
    background-color: {p.BLUE} !important; }}
  [data-testid="stSlider"] div[role="group"] > div:not([data-testid])
    > div:not([data-testid]) {{ background-color: {p.BLUE} !important; }}
  [data-testid="stSliderThumbValue"] {{
    color: {p.BLUE} !important; border-color: {p.BLUE} !important; }}
  .react-aria-SelectionIndicator {{
    background-color: {p.BLUE} !important; color: {p.BLUE} !important;
    border-color: {p.BLUE} !important; }}
  [data-testid="stTab"] {{ color: {p.MUTED} !important; }}
  [data-testid="stCheckbox"] svg {{ color: {p.BLUE}; }}

  /* ==================================================================
     Compact layer
     ==================================================================
     The reference dashboards fit their whole argument on one screen. That
     is a budget problem: at 1920x980 with the header removed there are
     roughly 900 usable pixels, and nine cards plus a KPI rail have to live
     inside it. Everything below buys vertical space back. */

  .block-container, .stMainBlockContainer {{
    padding-top: 0.55rem !important; padding-bottom: 0.35rem !important;
    max-width: 1880px !important; }}

  /* Streamlit's default column gutter is 1rem; on a nine-card grid that is
     ~60px of pure air. */
  [data-testid="stHorizontalBlock"] {{ gap: 0.5rem !important; }}
  [data-testid="stVerticalBlock"] {{ gap: 0.4rem !important; }}
  [data-testid="stElementContainer"] {{ margin-bottom: 0 !important; }}

  /* Card: the unit the grid is built from. Chart titles live INSIDE the
     figure (see mini() in app.py) because Streamlit collapses a markdown
     container above a chart to a few pixels and the plot paints over it. */
  .card {{
    background: {p.SURFACE}; border: 1px solid {p.BORDER};
    border-radius: 10px; padding: 6px 8px 2px 8px; height: 100%; }}
  .card-t {{
    font-size: 11.5px; font-weight: 650; color: {p.INK};
    letter-spacing: -0.005em; margin: 0 0 2px 2px; }}

  /* ==================================================================
     Water motif
     ==================================================================
     Restrained on purpose: a tinted surface, a droplet mark on the brand,
     and one wave rule under the header. Nothing animated, nothing that
     competes with the data for attention. */

  .drop {{
    display:inline-block; width:9px; height:9px; margin-right:6px;
    border-radius: 50% 50% 50% 0; transform: rotate(-45deg);
    background: linear-gradient(140deg, {p.AQUA} 0%, {p.BLUE} 70%);
    vertical-align: baseline; }}

  .waverule {{
    height: 6px; margin: 2px 0 6px 0; border-radius: 3px;
    background-image: url("data:image/svg+xml;utf8,\
<svg xmlns='http://www.w3.org/2000/svg' width='56' height='6' viewBox='0 0 56 6'>\
<path d='M0 4 Q 7 0 14 4 T 28 4 T 42 4 T 56 4' fill='none' \
stroke='%23{p.BLUE[1:]}' stroke-opacity='0.45' stroke-width='1.4'/></svg>");
    background-repeat: repeat-x; background-position: left center; }}

  /* KPI rail: a droplet-marked label, one line of value, one of context. */
  .kpistrip {{ gap: 6px !important; margin: 0 0 6px 0 !important; }}
  [data-testid="stMarkdown"]:has(.kpistrip) {{ min-height: 80px !important; }}
  .kpi {{ padding: 7px 11px !important; border-radius: 9px !important; }}
  .kpi-v {{ font-size: 21px !important; }}
  .kpi-title {{ background:
    linear-gradient(135deg, {p.TILE_WASH} 0%, transparent 65%), {p.SURFACE} !important; }}

  /* ==================================================================
     Sidebar
     ================================================================== */
  section[data-testid="stSidebar"] {{ width: 232px !important; }}
  /* Enough to breathe, but 62px less than Streamlit's default header strip.
     Zero clipped the brand against the top edge. */
  section[data-testid="stSidebar"] .block-container {{
    padding: 0.85rem 0.85rem 0.35rem 0.85rem !important; }}

  /* Streamlit reserves a 60px header strip at the top of the sidebar plus a
     16px margin — 76px of nothing above the brand mark. Its only occupant is
     the collapse control, and that is position:fixed now, so the strip can
     go and the logo moves up into the space. */
  [data-testid="stSidebarHeader"] {{
    height: 0 !important; min-height: 0 !important;
    padding: 0 !important; margin: 0 !important; }}
  /* stSidebarContent is the real scroll container in this build — the
     .block-container rule does not match inside the sidebar — so the small
     top inset has to go here or the brand sits flush against the edge. */
  [data-testid="stSidebarContent"] {{ padding-top: 14px !important; }}
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
    gap: 0.35rem !important; }}
  .sb-h {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: {p.MUTED};
    margin: 12px 0 2px 0; }}
  .sb-brand {{ font-size: 14px; font-weight: 700; color: {p.INK};
              line-height: 1.2; margin-bottom: 1px; padding-right: 26px; }}
  .sb-sub {{ font-size: 10.5px; color: {p.MUTED}; line-height: 1.35; }}
  section[data-testid="stSidebar"] label p {{ font-size: 11.5px !important; }}

  /* ==================================================================
     Filter controls
     ==================================================================
     config.toml deliberately sets no [theme] — setting ANY theme key makes
     Streamlit resolve a concrete base and force its widget chrome AND its
     built-in Plotly theme to light even when the palette is dark. The price
     is that widget chrome keeps Streamlit's light defaults, so the selects
     rendered as white boxes with near-black text on a dark sidebar. These
     rules repaint them from the palette, and shrink them so they sit at the
     same weight as the 30px nav rows instead of towering over them. */

  .react-aria-ComboBox div[role="group"],
  [data-baseweb="select"] > div {{
    background: {p.SURFACE} !important;
    border: 1px solid {p.BORDER} !important;
    border-radius: 8px !important;
    color: {p.INK} !important; }}
  .react-aria-ComboBox input {{
    color: {p.INK} !important; background: transparent !important; }}
  .react-aria-ComboBox input::placeholder {{
    color: {p.MUTED} !important; opacity: 1 !important; }}
  .react-aria-ComboBox button svg,
  .react-aria-ComboBox svg {{
    color: {p.MUTED} !important; fill: {p.MUTED} !important; }}

  /* Compact. Height comes down from 40px to 30px and the type from 14 to 12,
     which is what stops the three filters dominating the rail. */
  /* min-height, not height. A fixed 30px clipped the multiselect's tag
     container (28px tall, 38px of content), which turned overflow:auto into a
     visible scrollbar inside the filter box. Letting it grow means chips
     expand the control instead of scrolling inside it. */
  section[data-testid="stSidebar"] .react-aria-ComboBox,
  section[data-testid="stSidebar"] .react-aria-ComboBox div[role="group"] {{
    min-height: 30px !important; height: auto !important; }}
  section[data-testid="stSidebar"] [data-testid="stMultiSelectTagsContainer"] {{
    overflow: visible !important; }}
  section[data-testid="stSidebar"] .react-aria-ComboBox input {{
    height: 28px !important; font-size: 12px !important;
    padding: 2px 7px !important; }}
  section[data-testid="stSidebar"] [data-testid="stSelectbox"],
  section[data-testid="stSidebar"] [data-testid="stMultiSelect"] {{
    margin-bottom: 1px !important; }}
  section[data-testid="stSidebar"] [data-testid="stSelectbox"] label,
  section[data-testid="stSidebar"] [data-testid="stMultiSelect"] label {{
    margin-bottom: 1px !important; }}

  /* The dropdown renders in a portal OUTSIDE the sidebar, so these cannot be
     scoped to it. */
  .react-aria-Popover, [role="listbox"], [data-baseweb="popover"] [role="listbox"] {{
    background: {p.SURFACE} !important;
    border: 1px solid {p.BORDER} !important;
    border-radius: 9px !important; }}
  [role="option"] {{
    color: {p.INK_2} !important; font-size: 12px !important; }}
  [role="option"]:hover, [role="option"][data-focused="true"],
  [role="option"][aria-selected="true"] {{
    background: {p.TILE_WASH} !important; color: {p.INK} !important; }}

  /* Filter panel ------------------------------------------------------
     The trigger is one row in the rail; the panel it opens renders in a
     portal, so it can be far wider than the 232px sidebar. */
  .pop-h {{ font-size: 11px; color: {p.MUTED}; margin: 0 0 6px 0;
            line-height: 1.4; }}
  .pop-s {{ font-size: 10px; font-weight: 700; letter-spacing: 0.08em;
            text-transform: uppercase; color: {p.BLUE};
            margin: 9px 0 1px 0; }}
  [data-testid="stPopoverBody"] {{
    min-width: 430px !important;
    background: {p.SURFACE} !important;
    border: 1px solid {p.BORDER} !important; border-radius: 11px !important; }}
  [data-testid="stPopoverBody"] [data-testid="stCheckbox"] label p {{
    font-size: 11.5px !important; color: {p.INK_2} !important;
    text-transform: none !important; letter-spacing: 0 !important;
    font-weight: 400 !important; }}
  [data-testid="stPopoverBody"] [data-testid="stCheckbox"] {{
    margin-bottom: -6px !important; }}
  /* The trigger itself sits at nav-row weight. */
  section[data-testid="stSidebar"] [data-testid="stPopover"] button {{
    width: 100% !important; height: 30px !important; min-height: 0 !important;
    justify-content: flex-start !important;
    border: 1px solid {p.BORDER} !important; border-radius: 8px !important;
    background: {p.SURFACE} !important; }}
  section[data-testid="stSidebar"] [data-testid="stPopover"] button p {{
    font-size: 12px !important; color: {p.INK} !important;
    font-weight: 500 !important; }}

  /* ==================================================================
     Tables
     ==================================================================
     st.dataframe renders to a <canvas>, so its colours come from Streamlit's
     static config and cannot follow a runtime light/dark choice — the
     schedule stayed dark whatever the dashboard was set to. These are plain
     HTML tables instead, so they wear the palette like everything else. */
  .tblwrap {{
    overflow: auto; border: 1px solid {p.BORDER}; border-radius: 10px;
    background: {p.SURFACE}; }}
  table.tbl {{
    width: 100%; border-collapse: separate; border-spacing: 0;
    font-size: 12px; color: {p.INK_2}; }}
  table.tbl thead th {{
    position: sticky; top: 0; z-index: 2;
    background: {p.TABLE_HEAD}; color: {p.INK};
    font-weight: 650; font-size: 11px; text-align: left;
    padding: 8px 10px; white-space: nowrap;
    border-bottom: 1px solid {p.BORDER}; }}
  table.tbl td {{
    padding: 6px 10px; border-bottom: 1px solid {p.GRID};
    white-space: nowrap; }}
  table.tbl td.num {{
    text-align: right; font-variant-numeric: tabular-nums; }}
  table.tbl tbody tr:hover td {{ background: {p.TILE_WASH}; }}
  table.tbl tbody tr:last-child td {{ border-bottom: none; }}

  /* The progress column, in palette blue rather than Streamlit's alert red. */
  .tbar {{
    display: inline-block; width: 74px; height: 7px; border-radius: 4px;
    background: {p.GRID}; margin-right: 8px; vertical-align: middle; }}
  /* Red, from the status palette rather than a new hue. It reads as urgency
     here, which is what a priority score means — but note it is a FLAT red on
     every row, so it marks the column as "this is the priority" rather than
     encoding severity. Banding it by score would do that; say so if wanted. */
  .tbar-f {{
    display: block; height: 100%; border-radius: 4px; background: {p.CRITICAL}; }}
  .tbar-v {{ font-variant-numeric: tabular-nums; }}

  /* Reserve the table's height on its container, or the element after it
     (the download button) paints over the last rows. */
  [data-testid="stMarkdown"]:has(.tbl-s) {{ min-height: 312px !important; }}
  [data-testid="stMarkdown"]:has(.tbl-m) {{ min-height: 432px !important; }}
  [data-testid="stMarkdown"]:has(.tbl-l) {{ min-height: 542px !important; }}

  /* Download and other main-pane buttons. Streamlit paints these from its
     static config, so without this they stay light even in dark mode — the
     same reason the tables needed rewriting. */
  .stMain [data-testid="stDownloadButton"] button,
  .stMain [data-testid="stBaseButton-secondary"],
  .stMain [data-testid="stBaseButton-primary"] {{
    background: {p.SURFACE} !important;
    color: {p.INK} !important;
    border: 1px solid {p.BORDER} !important;
    border-radius: 8px !important; }}
  .stMain [data-testid="stDownloadButton"] button p,
  .stMain [data-testid="stBaseButton-secondary"] p {{
    color: {p.INK} !important; font-size: 12.5px !important; }}
  .stMain [data-testid="stDownloadButton"] button:hover,
  .stMain [data-testid="stBaseButton-secondary"]:hover {{
    background: {p.TILE_WASH} !important;
    border-color: {p.BLUE} !important; }}

  /* ==================================================================
     Element toolbars
     ==================================================================
     Streamlit floats a fullscreen / download toolbar over images and charts
     on hover. On the brand mark it reads as a stray button sitting on the
     logo, so it is removed there and left on the charts, where expanding a
     figure is genuinely useful. */
  [data-testid="stImage"] [data-testid="StyledFullScreenButton"],
  [data-testid="stImage"] [data-testid="stElementToolbar"],
  section[data-testid="stSidebar"] [data-testid="stElementToolbar"],
  .stMain > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]:first-child
    [data-testid="stImage"] [data-testid="stElementToolbar"] {{
    display: none !important; }}

  /* ==================================================================
     Sidebar navigation
     ==================================================================
     Rows are Streamlit buttons rather than a radio group: the sections need
     headings between them, and one radio cannot carry those. Each row is
     preceded by an empty .nav-row marker div whose only job is to give the
     ACTIVE row a hook for its left accent bar — Streamlit does not expose a
     class on the button itself. The marker is pulled down over the button
     with a negative margin so the two occupy the same line. */

  .nav-h {{
    font-size: 9.5px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {p.BLUE};
    margin: 13px 0 3px 2px; opacity: 0.85; }}

  .nav-row {{ height: 0; }}
  .nav-row.nav-on {{
    height: 0; border-left: 3px solid {p.BLUE};
    margin: 0 0 -30px -8px; padding-top: 30px; }}

  /* The rows themselves: full width, left-aligned, no button chrome. */
  section[data-testid="stSidebar"] .stButton > button {{
    width: 100% !important; justify-content: flex-start !important;
    text-align: left !important;
    padding: 4px 9px !important; min-height: 0 !important; height: 30px !important;
    border: none !important; border-radius: 7px !important;
    background: transparent !important; box-shadow: none !important; }}
  /* The label sits in an inner flex div that centres itself; the button's own
     justify-content does not reach it. */
  section[data-testid="stSidebar"] .stButton > button > div {{
    justify-content: flex-start !important; width: 100% !important; }}
  section[data-testid="stSidebar"] .stButton > button p {{
    font-size: 12px !important; font-weight: 500 !important;
    color: {p.INK_2} !important; margin: 0 !important; }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {p.TILE_WASH} !important; }}
  section[data-testid="stSidebar"] .stButton > button:hover p {{
    color: {p.INK} !important; }}

  /* Active row. Streamlit paints a primary button with its accent fill; this
     replaces that with a tinted wash so the sidebar stays quiet. */
  section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: {p.TILE_WASH} !important; }}
  section[data-testid="stSidebar"] .stButton > button[kind="primary"] p {{
    color: {p.INK} !important; font-weight: 650 !important; }}

  /* Navigation is a long list; let it scroll inside the rail rather than
     pushing the appearance control out of reach. */
  section[data-testid="stSidebar"] .block-container {{
    max-height: 100vh; overflow-y: auto; }}

  /* ==================================================================
     Cards
     ==================================================================
     Real containers (st.container(border=True)) rather than a plot whose
     paper rectangle pretends to be a card. The difference is padding: a
     figure drawn edge-to-edge inside a border reads as a pasted-in picture,
     which was most of what made this look unfinished next to a reference
     dashboard. */
  /* Streamlit puts the border on the stVerticalBlock itself in this build,
     not on a wrapper — the wrapper test id does not exist here, so the earlier
     rule matched nothing and every card was still wearing Streamlit's default
     grey hairline. Selecting on the presence of our own card header is both
     precise and self-describing: this is a block that contains a card title. */
  .stMain [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-t) {{
    background: {p.SURFACE} !important;
    border: 1px solid {p.BORDER} !important;
    border-radius: 12px !important;
    padding: 11px 13px 7px 13px !important; }}

  /* The card's own title row. Sized above the axis labels so the hierarchy
     inside a card is title > marks > axis, not three flavours of small. */
  .card-t {{
    font-size: 13px !important; font-weight: 650 !important;
    color: {p.INK} !important; letter-spacing: -0.008em;
    margin: 0 0 1px 0 !important; line-height: 1.25; }}
  .card-s {{
    font-size: 10.5px !important; color: {p.MUTED} !important;
    margin: 0 0 3px 0 !important; line-height: 1.3; }}

  /* Streamlit sizes a markdown container from its pre-layout content, so a
     two-line card header collapsed to roughly half its height and the chart
     below painted over the subtitle. Reserve the space explicitly. */
  [data-testid="stMarkdown"]:has(.card-t) {{ min-height: 37px !important; }}

  /* Hero stat, for the one card per screen that is a number rather than a
     plot. Proportional figures — tabular-nums makes a large number look
     loose. */
  .hero-v {{
    font-size: 40px; font-weight: 660; color: {p.INK};
    line-height: 1.02; letter-spacing: -0.025em; margin: 2px 0 0 0; }}
  .hero-u {{
    font-size: 15px; font-weight: 500; color: {p.INK_2}; margin-left: 3px; }}
  .hero-s {{
    font-size: 11px; color: {p.INK_2}; margin-top: 5px; line-height: 1.4; }}

  /* ==================================================================
     Sidebar collapse control
     ==================================================================
     Streamlit ships it unstyled and near-invisible against a tinted surface,
     so the sidebar looked like it had no way to close. */
  [data-testid="stSidebarCollapseButton"] {{
    position: fixed !important; top: 9px !important; left: 196px !important;
    z-index: 1002 !important; }}
  [data-testid="stSidebarCollapseButton"] button {{
    background: {p.SURFACE} !important;
    border: 1px solid {p.BORDER} !important;
    border-radius: 8px !important; width: 28px !important; height: 28px !important; }}
  [data-testid="stSidebarCollapseButton"] button:hover {{
    background: {p.TILE_WASH} !important; }}
  /* The icon is a Material FONT GLYPH in a span, not an SVG, so styling svg
     did nothing and it kept Streamlit's rgba(49,51,63,0.6) — invisible on
     both of these surfaces. The button looked like an empty box. */
  [data-testid="stSidebarCollapseButton"] svg,
  [data-testid="stSidebarCollapseButton"] span,
  [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"] {{
    color: {p.INK_2} !important; fill: {p.INK_2} !important;
    opacity: 1 !important; }}
  [data-testid="stSidebarCollapseButton"] button:hover span {{
    color: {p.INK} !important; }}
  /* Same for the expand control, which uses the same glyph markup. */
  [data-testid="stExpandSidebarButton"] span,
  [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
    color: {p.INK_2} !important; opacity: 1 !important; }}

  /* A visible scrollbar arrow in the sidebar reads as a stray caret; make the
     whole bar quiet instead. */
  section[data-testid="stSidebar"] *::-webkit-scrollbar {{ width: 7px; }}
  section[data-testid="stSidebar"] *::-webkit-scrollbar-button {{ display: none; }}
  section[data-testid="stSidebar"] *::-webkit-scrollbar-track {{ background: transparent; }}
  section[data-testid="stSidebar"] *::-webkit-scrollbar-thumb {{
    background: {p.BASELINE}; border-radius: 4px; }}

  /* Sub-tabs tighter still inside a dense grid. */
  .stTabs [data-testid="stTab"] {{ height: 34px !important; padding: 0 13px !important; }}
  .stTabs [data-testid="stTab"] p {{ font-size: 12.5px !important; }}
  .stTabs .stTabs [data-testid="stTab"] {{ height: 28px !important; padding: 0 10px !important; }}
  .stTabs .stTabs [data-testid="stTab"] p {{ font-size: 11.5px !important; }}

  .callout {{ padding: 9px 13px !important; margin: 2px 0 8px 0 !important;
             font-size: 12px !important; line-height: 1.5 !important; }}
  .caption {{ font-size: 11px !important; margin: -2px 0 6px 0 !important; }}
  h4 {{ font-size: 15px !important; margin: 2px 0 4px 0 !important; }}
  h5, h6 {{ font-size: 12.5px !important; margin: 2px 0 3px 0 !important; }}
</style>
"""


def resolve_mode(preference: str, detected: str) -> str:
    """`preference` is the sidebar override; `detected` comes from the browser
    or OS via st.context.theme. Auto follows the system."""
    if preference == "Light":
        return "light"
    if preference == "Dark":
        return "dark"
    return detected if detected in PALETTES else "light"
