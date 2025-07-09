import os
import matplotlib.colors as mcolors
import colorsys
from pathlib import Path


def darken_color(hex_color, sat_factor=1.2, light_factor=0.8):
    """Adjusts a color to be more visible as a line by increasing saturation and reducing lightness."""
    rgb = mcolors.hex2color(hex_color)  # Convert to RGB (0-1)
    h, l, s = colorsys.rgb_to_hls(*rgb)  # Convert to HLS

    # Adjust saturation and lightness
    new_s = min(1, s * sat_factor)  # Boost saturation
    new_l = max(0, l * light_factor)  # Reduce lightness slightly

    new_rgb = colorsys.hls_to_rgb(h, new_l, new_s)  # Convert back to RGB
    return mcolors.to_hex(new_rgb)  # Convert to hex


def fill_color(hex_color):
    return darken_color(hex_color, sat_factor=0.85, light_factor=0.88)


def configure_kernel_cache_dir():
    cache_dir = Path(os.environ.get('TILUS_ARTIFACT_CACHE_DIR', './cache'))
    cache_dir.mkdir(parents=True, exist_ok=True)

    # configure bitblas cache directory
    import bitblas.cache
    bitblas.cache.set_database_path(str(cache_dir / 'bitblas'))
    bitblas.module.BITBLAS_DATABASE_PATH = str(cache_dir / 'bitblas')

    # configure triton cache directory
    os.environ['TRITON_CACHE_DIR'] = str(cache_dir / 'triton')

    # configure mutis cache directory
    import mutis
    mutis.option.cache_dir(str(cache_dir / 'mutis'))

configure_kernel_cache_dir()
