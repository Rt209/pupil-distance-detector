from __future__ import annotations

import argparse

PIPELINE_NAMES = ("v1", "v2", "v3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect pupils and measure inter-pupil distance.")
    parser.add_argument("--input", required=True, help="Path to the input face image.")
    parser.add_argument("--output", required=True, help="Path to save the annotated output image.")
    parser.add_argument(
        "--pipeline",
        default="v1",
        choices=PIPELINE_NAMES,
        help="Pipeline name to run.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Load the input image as grayscale.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from pupil_distance_detector.pipelines.factory import build_pipeline
    from pupil_distance_detector.utils.image_io import load_image, save_image

    image = load_image(args.input, grayscale=args.grayscale)
    pipeline = build_pipeline(args.pipeline)
    result = pipeline.run(image)
    save_image(args.output, result.annotated_image)

    print(f"Pipeline: {result.pipeline_name}")
    print(f"Left pupil center: {result.left_candidate.center}")
    print(f"Right pupil center: {result.right_candidate.center}")
    print(f"Distance: {result.distance_px:.2f} px")
    print(f"Saved annotated output to {args.output}")


if __name__ == "__main__":
    main()
