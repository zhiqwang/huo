"""Sphinx configuration for the Huo documentation."""

from pathlib import Path

import pydata_sphinx_theme

project = "Huo Documentation"
language = "en"
root_doc = "index"
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["_templates/**", "_static/**"]
myst_enable_extensions = ["attrs_inline", "colon_fence", "dollarmath"]
myst_heading_anchors = 6

html_theme = "pytorch_sphinx_theme2"
html_title = project
html_baseurl = "https://zhiqwang.github.io/huo/"
templates_path = ["_templates"]
html_theme_options = {
    "logo": {"text": project},
    "navbar_center": [],
    "navbar_end": ["theme-switcher"],
    "navbar_persistent": ["search-button"],
    "show_nav_level": 1,
    "navigation_depth": 3,
    "collapse_navigation": True,
    "show_prev_next": True,
    "display_version": False,
    "show_pytorch_org_link": False,
    "show_lf_header": False,
    "show_lf_footer": False,
    "article_header_end": [],
    "article_footer_items": [],
    "secondary_sidebar_items": ["page-toc"],
    "use_edit_page_button": False,
    "llm_disabled": True,
    "github_url": "https://github.com/zhiqwang/huo",
}
html_sidebars = {"**": ["huo-nav.html"]}
html_show_sourcelink = False
# The pinned theme omits its webfonts; use the assets from PyData's theme.
fontawesome_assets = next(
    Path(pydata_sphinx_theme.__file__).parent.glob(
        "theme/pydata_sphinx_theme/static/vendor/fontawesome/*/webfonts"
    )
).parent
html_static_path = ["_static", str(fontawesome_assets)]
html_css_files = ["custom.css"]
html_js_files = [
    ("js/jquery.min.js", {"defer": "defer"}),
    ("js/theme.js", {"defer": "defer"}),
]
mathjax3_config = {"tex": {"tags": "ams"}}
