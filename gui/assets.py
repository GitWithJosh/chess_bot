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
