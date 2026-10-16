"""Shared palette and MUI theme for the react client.

Matches the vtk-history look: IBM Plex, teal accent, light surfaces.
"""

CANVAS = "#F3F5F8"
SURFACE = "#FFFFFF"
INK = "#172033"
MUTED = "#5A6474"
LINE = "#DCE1E8"
ACCENT = "#0A7C8C"
ACCENT_TINT = "#E3F2F4"

SANS = "'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
    "&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
)
# mui.Icon renders a ligature from this font; trame-mui ships @mui/material
# only, so @mui/icons-material components are not available.
ICON_FONT = "https://fonts.googleapis.com/icon?family=Material+Icons"

THEME = {
    "palette": {
        "mode": "light",
        "primary": {"main": ACCENT},
        "secondary": {"main": "#6E56CF"},
        "success": {"main": "#1F883D"},
        "error": {"main": "#C4314B"},
        "warning": {"main": "#B35900"},
        "background": {"default": CANVAS, "paper": SURFACE},
        "text": {"primary": INK, "secondary": MUTED},
        "divider": LINE,
    },
    "typography": {
        "fontFamily": SANS,
        "fontSize": 13,
        "button": {"textTransform": "none", "fontWeight": 500},
    },
    "shape": {"borderRadius": 6},
    "components": {
        "MuiButton": {"defaultProps": {"disableElevation": True}},
        "MuiButtonGroup": {"defaultProps": {"disableElevation": True}},
        "MuiPaper": {"styleOverrides": {"outlined": {"borderColor": LINE}}},
        "MuiChip": {
            "styleOverrides": {
                "root": {"borderRadius": "6px", "fontWeight": 500},
                "sizeSmall": {"height": "22px"},
            }
        },
        "MuiOutlinedInput": {
            "styleOverrides": {
                "root": {"backgroundColor": SURFACE},
                "notchedOutline": {"borderColor": LINE},
            }
        },
        "MuiListItemButton": {
            "styleOverrides": {
                "root": {
                    "borderRadius": "6px",
                    "&.Mui-selected": {"backgroundColor": ACCENT_TINT},
                }
            }
        },
        "MuiTooltip": {"defaultProps": {"arrow": True, "enterDelay": 400}},
    },
}
