import os
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
import tifffile


# Disable DecompressionBombError. WSI images are massive and will trigger Pillow limits.
Image.MAX_IMAGE_PIXELS = None


def convert_ndpi_to_tiff(
    ndpi_file_path,
    output_dir=None,
    output_tiff="he-raw.tiff",
    output_txt="pixel-size-raw.txt",
    target_level=0,
):
    """
    Converts an NDPI file to a TIFF image and extracts effective pixel size in micrometers.
    """
    if not os.path.exists(ndpi_file_path):
        raise FileNotFoundError(f"Could not find input NDPI: {ndpi_file_path}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(ndpi_file_path))
    os.makedirs(output_dir, exist_ok=True)

    output_tiff = os.path.join(output_dir, output_tiff)
    output_txt = os.path.join(output_dir, output_txt)

    print(f"Opening {ndpi_file_path}...")
    slide = openslide.OpenSlide(ndpi_file_path)

    try:
        try:
            base_mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            downsample_factor = slide.level_downsamples[target_level]
            effective_mpp = base_mpp_x * downsample_factor
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write(f"{effective_mpp:.6f}\n")
            print(f"Success: Wrote pixel size {effective_mpp:.6f} um to '{output_txt}'.")
        except KeyError:
            print("Warning: MPP physical size data not found in the NDPI metadata.")
            with open(output_txt, "w", encoding="utf-8") as f:
                f.write("Unknown\n")

        dimensions = slide.level_dimensions[target_level]
        print(
            f"Extracting level {target_level} with dimensions {dimensions[0]}x{dimensions[1]} pixels..."
        )

        img = slide.read_region((0, 0), target_level, dimensions).convert("RGB")

        print(f"Saving image to '{output_tiff}' (this may take a moment for large files)...")
        tifffile.imwrite(output_tiff, np.asarray(img), photometric="rgb")
        print("Done!")
    except Exception as exc:
        raise RuntimeError(f"NDPI image extraction failed for '{ndpi_file_path}': {exc}") from exc
    finally:
        slide.close()


def convert_ndpi_to_image(*args, **kwargs):
    """Backward-compatible alias used in notebooks and docs."""
    return convert_ndpi_to_tiff(*args, **kwargs)


def _has_he_raw(sample_out_dir):
    base = Path(sample_out_dir) / "he-raw"
    for suffix in (".png", ".ome.tif", ".tiff", ".tif", ".svs"):
        if base.with_suffix(suffix).exists():
            return True
    return False


def convert_ndpi_with_fallback(ndpi_path, sample_out_dir, target_level):
    # Retry at deeper pyramid levels when full-resolution extraction is too large.
    for level in range(target_level, target_level + 5):
        try:
            convert_ndpi_to_tiff(
                ndpi_file_path=ndpi_path,
                output_dir=sample_out_dir,
                output_tiff="he-raw.tiff",
                output_txt="pixel-size-raw.txt",
                target_level=level,
            )
            if _has_he_raw(sample_out_dir) and (
                Path(sample_out_dir) / "pixel-size-raw.txt"
            ).exists():
                if level != target_level:
                    print(
                        f"[WARN] NDPI extraction retried at target level {level} "
                        f"(requested {target_level})."
                    )
                return
            raise RuntimeError("Conversion finished but expected output files were not created.")
        except Exception as exc:
            if level == target_level + 4:
                raise RuntimeError(
                    f"NDPI conversion failed after retries from level {target_level} to {level}: {exc}"
                ) from exc
            print(
                f"[WARN] NDPI conversion failed at level {level} for '{ndpi_path}'. "
                f"Retrying deeper level. Reason: {exc}"
            )
