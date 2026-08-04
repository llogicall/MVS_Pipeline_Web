"""Project-compatible MVSTTA inference entrypoint."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.mvs_mvstta import run_depth_estimation, run_point_fusion


def build_parser():
    parser = argparse.ArgumentParser(description="MVSTTA depth-map testing")
    parser.add_argument("--dataset", default="general_eval", choices=["general_eval", "aerial_mvs", "dtu_yao", "blendedmvs"])
    parser.add_argument("--testpath", required=True)
    parser.add_argument("--testpath_single_scene", default=None)
    parser.add_argument("--testlist", default="all")
    parser.add_argument("--loadckpt", required=True)
    parser.add_argument("--outdir", default="./result/depths")
    parser.add_argument("--num_view", "--nviews", dest="num_view", type=int, default=5)
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--max_h", type=int, default=512)
    parser.add_argument("--max_w", type=int, default=640)
    parser.add_argument("--num_worker", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.9)
    parser.add_argument("--thres_view", type=int, default=5)
    parser.add_argument("--tta_lr", type=float, default=5e-5)
    parser.add_argument("--tta_steps", type=int, default=1)
    parser.add_argument("--ds_mode", default="reprojectionv5")
    parser.add_argument("--no_gpu", dest="use_gpu", action="store_false")
    parser.add_argument("--skip_fusion", action="store_true")
    parser.set_defaults(use_gpu=True)
    return parser


def main():
    args = build_parser().parse_args()
    testpath = args.testpath
    if args.testpath_single_scene:
        scene_path = Path(args.testpath_single_scene).resolve()
        testpath = str(scene_path.parent)

    log = print
    run_depth_estimation(
        testpath=testpath,
        outdir=args.outdir,
        checkpoint=args.loadckpt,
        log=log,
        dataset=args.dataset,
        num_view=args.num_view,
        numdepth=args.numdepth,
        interval_scale=args.interval_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        num_worker=args.num_worker,
        tta_lr=args.tta_lr,
        tta_steps=args.tta_steps,
        ds_mode=args.ds_mode,
        use_gpu=args.use_gpu,
    )
    if not args.skip_fusion:
        run_point_fusion(
            testpath=testpath,
            outdir=args.outdir,
            conf=args.conf,
            thres_view=args.thres_view,
            num_worker=args.num_worker,
            log=log,
        )


if __name__ == "__main__":
    main()
