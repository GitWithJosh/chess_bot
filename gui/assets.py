"""Image loading shared by the menu and the game screen.

Two things the plain pygame calls do not do for us.

Scaling with transform.scale drops pixels rather than averaging them, which
shows badly here because every asset is downscaled a long way, 128 px pieces to
70 and 512 px icons to 50. smoothscale interpolates instead.

The icons are solid black on transparency, so on the dark panels they were very
hard to make out. Tinting lifts them to a readable grey.
"""

import os

import pygame

ICON_TINT = (215, 215, 215)

# pygame's bundled freesansbold hints badly below about 30 and renders far
# smaller than its nominal size, 14 px tall at size 20, which is what made digits
# uneven and l against i unreadable. Prefer a real UI face, regular and bold.
# match_font is not used because it fuzzy-matches, returning Segoe UI Light for
# "segoeui" and Arial Narrow for "arial".
_FONT_CANDIDATES = [
    ("segoeui.ttf", "segoeuib.ttf"),
    ("tahoma.ttf", "tahomabd.ttf"),
    ("verdana.ttf", "verdanab.ttf"),
    ("arial.ttf", "arialbd.ttf"),
]

_font_cache: dict[tuple[int, bool], pygame.font.Font] = {}
_font_files: tuple[str | None, str | None] | None = None


def _resolve_font_files() -> tuple[str | None, str | None]:
    """First candidate family present, as (regular, bold) paths."""
    global _font_files
    if _font_files is not None:
        return _font_files
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for regular, bold in _FONT_CANDIDATES:
        r_path = os.path.join(fonts_dir, regular)
        if os.path.exists(r_path):
            b_path = os.path.join(fonts_dir, bold)
            _font_files = (r_path, b_path if os.path.exists(b_path) else r_path)
            return _font_files
    _font_files = (None, None)      # falls back to the pygame default
    return _font_files


def ui_font(size: int, bold: bool = False) -> pygame.font.Font:
    """A cached UI font. Sizes are in real points, not the pygame default's scale."""
    key = (size, bold)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    regular, bold_path = _resolve_font_files()
    path = bold_path if bold else regular
    try:
        font = pygame.font.Font(path, size)
    except (OSError, pygame.error):
        font = pygame.font.Font(None, size)
        font.set_bold(bold)
    _font_cache[key] = font
    return font

_PIECES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "pieces-basic-png")


def asset_path(filename: str) -> str:
    return os.path.join(_PIECES_DIR, filename)


def load_image(filename: str, size: tuple[int, int],
               tint: tuple[int, int, int] | None = None) -> pygame.Surface | None:
    """Load, scale smoothly and optionally tint. None if the file is missing.

    convert_alpha needs a display mode, which both callers have set by the time
    they load anything.
    """
    path = asset_path(filename)
    if not os.path.exists(path):
        return None
    try:
        img = pygame.image.load(path).convert_alpha()
        img = pygame.transform.smoothscale(img, size)
        if tint is not None:
            # RGB_ADD leaves alpha alone, so a black glyph becomes exactly the
            # tint colour and the transparent surround stays transparent.
            img.fill(tint, special_flags=pygame.BLEND_RGB_ADD)
        return img
    except pygame.error:
        return None


def load_icon(filename: str, size: tuple[int, int]) -> pygame.Surface | None:
    """A UI icon, tinted for contrast against the dark panels."""
    return load_image(filename, size, tint=ICON_TINT)
