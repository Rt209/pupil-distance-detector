from __future__ import annotations

import argparse

PIPELINE_NAMES = ("v1", "v2", "v3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="瞳孔距離偵測工具")
    parser.add_argument("--input", required=True, help="輸入影像路徑")
    parser.add_argument("--output", required=True, help="輸出成果影像路徑")
    parser.add_argument(
        "--pipeline",
        default="v1",
        choices=PIPELINE_NAMES,
        help="選擇要執行的 pipeline",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="以灰階模式讀取輸入影像",
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
    print(f"左瞳孔中心: {result.left_candidate.center}")
    print(f"右瞳孔中心: {result.right_candidate.center}")
    print(f"瞳孔中心距離: {result.distance_px:.2f} px")
    print(f"已輸出成果影像: {args.output}")


if __name__ == "__main__":
    main()
