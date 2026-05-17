"""
Generate 28x28 PNG images of letters, centered, with a rotation angle parameter.
Uses only Pillow (PIL) — install with: pip install Pillow
"""

from PIL import Image, ImageDraw, ImageFont
import os
import math


def generate_letter_image(
    letter: str,
    angle: float = 0.0,
    output_path: str = None,
    font_size: int = 50,
    bg_color: int = 255,      # 255 = white background
    fg_color: int = 0,        # 0 = black letter
    image_size: int = 64,
) -> Image.Image:
    """
    Generate a 28x28 grayscale PNG image of a single letter, centered and rotated.

    Args:
        letter:      Single character to render.
        angle:       Rotation angle in degrees (counter-clockwise).
        output_path: If given, saves the image to this path.
        font_size:   Font size used to draw the letter (default 22 fits well in 28px).
        bg_color:    Background intensity 0-255 (default 0 = black, like MNIST).
        fg_color:    Foreground intensity 0-255 (default 255 = white).
        image_size:  Width and height of the output image (default 28).

    Returns:
        PIL Image object (mode 'L', 28×28).
    """
    if len(letter) != 1:
        raise ValueError(f"Expected a single character, got: {repr(letter)}")

    # --- draw on a larger canvas so rotation doesn't clip corners ---
    pad = image_size * 2          # generous padding
    canvas = pad + image_size     # e.g. 84 for 28-px output
    big = Image.new("L", (canvas, canvas), color=bg_color)
    draw = ImageDraw.Draw(big)

    # Try to load a clean monospace font; fall back to default if unavailable
    font = None
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Courier.ttc",          # macOS
        "C:/Windows/Fonts/cour.ttf",                  # Windows
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=font_size)
                break
            except Exception:
                pass
    if font is None:
        # PIL built-in bitmap font — smaller but always available
        font = ImageFont.load_default()

    # Measure the letter so we can center it precisely
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (canvas - text_w) / 2 - bbox[0]
    y = (canvas - text_h) / 2 - bbox[1]
    draw.text((x, y), letter, fill=fg_color, font=font)

    # Rotate around the center of the canvas
    big = big.rotate(angle, resample=Image.BICUBIC, center=(canvas / 2, canvas / 2))

    # Crop the central 28×28 region
    left = pad // 2
    top  = pad // 2
    img = big.crop((left, top, left + image_size, top + image_size))

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        img.save(output_path, format="PNG")
        print(f"Saved: {output_path}")

    return img


def generate_alphabet(
    angles: list = None,
    output_dir: str = "letters",
    letters: str = None,
    **kwargs,
):
    """
    Generate images for every letter (or a custom set) at one or more angles.

    Args:
        angles:     List of rotation angles in degrees. Defaults to [0].
        output_dir: Directory to save images.
        letters:    String of characters to render. Defaults to A-Z.
        **kwargs:   Forwarded to generate_letter_image().
    """
    if angles is None:
        angles = [0]
    if letters is None:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for letter in letters:
        for angle in angles:
            filename = f"{letter}_{int(angle):03d}deg.png"
            path = os.path.join(output_dir, filename)
            generate_letter_image(letter, angle=angle, output_path=path, **kwargs)


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Single letter at a specific angle
    img = generate_letter_image("A", angle=30, output_path="A_30deg.png")
    print(f"Image size: {img.size}, mode: {img.mode}")

    # 2. Generate A–Z at 0°, 45°, and 90°
    generate_alphabet(
        letters="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        angles=[0, 45, 90],
        output_dir="Letters",
    )
    print("Done — images saved in Letters/")