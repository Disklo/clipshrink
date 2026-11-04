LOSSY_FORMATS = ("JPEG", "WEBP", "AVIF", "JXL", "JP2", "HEIC", "TIFF")

# no quality knob, palette optimizes PNG size so conversion can overflow
PALETTE_CORRECTION_FORMATS = ("GIF", "BMP", "VIPS")

ANIMATION_SOURCE_FORMATS = (
    "GIF", "WEBP", "AVIF", "JXL", "HEIC", "HEIF", "TIFF", "PNG",
)

# APNG needs the explicit magick coder, the still PNG pipeline owns it
ANIMATED_TARGET_FORMATS = ("GIF", "WEBP", "AVIF", "JXL", "HEIC", "HEIF", "TIFF")

ANIMATED_LOSSY_FORMATS = ("WEBP", "AVIF", "JXL", "HEIC", "HEIF", "TIFF")

# Keyed by target format: shrinkability belongs to the output codec, so -F
# conversions must not inherit the input's fraction. Values are --slow
# --ssim-target 0.95 medians over 29 stock photos, rounded up.
DEFAULT_RELATIVE_FRACTION = 0.6
DEFAULT_TARGET_FRACTIONS: dict[str, float] = {
    "JPEG": 0.42,
    "WEBP": 0.41,
    "AVIF": 0.31,
    "JXL": 0.41,
    "JP2": 0.15,
    "HEIC": 0.19,
    "TIFF": 0.05,
}


def default_target_fraction(target_format: str | None) -> float:
    if target_format is None:
        return DEFAULT_RELATIVE_FRACTION
    return DEFAULT_TARGET_FRACTIONS.get(target_format.upper(),
                                        DEFAULT_RELATIVE_FRACTION)
