import argparse
import glob
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
import tifffile

from s2omics.p1_histology_preprocess import histology_preprocess
from s2omics.p2_superpixel_quality_control import superpixel_quality_control
from s2omics.p3_feature_extraction import histology_feature_extraction


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


def _has_he_raw(sample_out_dir):
    base = Path(sample_out_dir) / "he-raw"
    for suffix in (".jpg", ".png", ".ome.tif", ".tiff", ".tif", ".svs"):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch pipeline for NDPI -> TIFF + S2Omics steps 1-3. "
            "Designed for cluster usage across many images."
        )
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default=None,
        help="Glob pattern for NDPI files, e.g. '/data/**/*.ndpi'.",
    )
    parser.add_argument(
        "--input-list",
        type=str,
        default=None,
        help="Text file with one NDPI path per line.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        required=True,
        help=(
            "Output base directory. Each image will use a subfolder named after "
            "the NDPI filename stem."
        ),
    )
    parser.add_argument(
        "--target-level",
        type=int,
        default=0,
        help="NDPI pyramid level for conversion (0 = highest resolution).",
    )
    parser.add_argument(
        "--foundation-model",
        type=str,
        default="uni",
        choices=["uni", "virchow", "gigapath"],
        help="Foundation model used in feature extraction.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="./checkpoints/uni/",
        help="Checkpoint folder containing pytorch_model.bin for selected model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for step 3, e.g. 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction.",
    )
    parser.add_argument(
        "--down-samp-step",
        type=int,
        default=10,
        help="Down-sampling step used in step 3.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers in step 3.",
    )
    parser.add_argument(
        "--show-image",
        action="store_true",
        help="Show matplotlib previews (usually disabled for clusters).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sample if final step-3 output already exists.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort whole batch when one sample fails.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap on number of images processed after filtering/splitting.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="0-based task index for array job splitting.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Total array tasks for splitting images across cluster jobs.",
    )
    return parser.parse_args()


def read_input_list(input_list_path):
    paths = []
    with open(input_list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            paths.append(line)
    return paths


def collect_inputs(args):
    if not args.input_glob and not args.input_list:
        raise ValueError("Provide at least one of --input-glob or --input-list.")

    files = []
    if args.input_glob:
        files.extend(glob.glob(args.input_glob, recursive=True))
    if args.input_list:
        files.extend(read_input_list(args.input_list))

    # Keep ordering deterministic and unique.
    files = sorted({os.path.abspath(p) for p in files})

    if not files:
        raise ValueError("No NDPI files found from provided inputs.")

    missing = [p for p in files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} input files do not exist. First missing: {missing[0]}"
        )

    return files


def split_for_task(files, task_id=None, num_tasks=None):
    if task_id is None and num_tasks is None:
        return files
    if task_id is None or num_tasks is None:
        raise ValueError("Use --task-id and --num-tasks together.")
    if num_tasks <= 0:
        raise ValueError("--num-tasks must be > 0.")
    if task_id < 0 or task_id >= num_tasks:
        raise ValueError("--task-id must satisfy 0 <= task_id < num_tasks.")
    return [p for idx, p in enumerate(files) if idx % num_tasks == task_id]


def already_finished(sample_out_dir, foundation_model, down_samp_step):
    target = (
        Path(sample_out_dir)
        / "S2Omics_output"
        / "pickle_files"
        / f"{foundation_model}_embeddings_downsamp_{down_samp_step}_part_0.pickle"
    )
    return target.exists()


def process_one(ndpi_path, args):
    sample_name = Path(ndpi_path).stem
    sample_out_dir = os.path.join(args.work_dir, sample_name)
    os.makedirs(sample_out_dir, exist_ok=True)

    if args.skip_existing and already_finished(
        sample_out_dir, args.foundation_model, args.down_samp_step
    ):
        print(f"[SKIP] {sample_name}: step-3 outputs already found")
        return

    print(f"[START] {sample_name}")

    # Convert NDPI into he-raw.tiff and pixel-size-raw.txt in sample folder.
    convert_ndpi_with_fallback(
        ndpi_path=ndpi_path,
        sample_out_dir=sample_out_dir,
        target_level=args.target_level,
    )

    prefix = sample_out_dir.rstrip("/") + "/"
    save_folder = os.path.join(sample_out_dir, "S2Omics_output")

    # Step 1
    histology_preprocess(prefix, show_image=args.show_image)

    # Step 2
    superpixel_quality_control(prefix, save_folder, show_image=args.show_image)

    # Step 3
    histology_feature_extraction(
        prefix,
        save_folder,
        foundation_model=args.foundation_model,
        ckpt_path=args.ckpt_path,
        device=args.device,
        batch_size=args.batch_size,
        down_samp_step=args.down_samp_step,
        num_workers=args.num_workers,
    )

    print(f"[DONE]  {sample_name}")


def main():
    args = parse_args()

    files = collect_inputs(args)
    files = split_for_task(files, args.task_id, args.num_tasks)

    if args.max_images is not None:
        files = files[: args.max_images]

    if not files:
        print("No files assigned to this run after filtering/splitting.")
        return

    print(f"Total assigned files: {len(files)}")
    print(f"Output base directory: {os.path.abspath(args.work_dir)}")
    print(f"Foundation model: {args.foundation_model}")
    print(f"Device: {args.device}")

    failed = []
    for ndpi_path in files:
        try:
            process_one(ndpi_path, args)
        except Exception as exc:
            print(f"[FAIL]  {ndpi_path}")
            print(f"Reason: {exc}")
            traceback.print_exc()
            failed.append(ndpi_path)
            if args.stop_on_error:
                break

    print("\nBatch finished.")
    print(f"Success: {len(files) - len(failed)}")
    print(f"Failed:  {len(failed)}")

    if failed:
        print("Failed files:")
        for p in failed:
            print(p)
        sys.exit(1)


if __name__ == "__main__":
    main()