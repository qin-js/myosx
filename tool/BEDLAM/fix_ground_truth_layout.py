import argparse
import shutil
from pathlib import Path


def move_path(src, dst, apply, overwrite=False):
    if not src.exists():
        return "missing", src, dst
    if dst.exists():
        if not overwrite:
            return "exists", src, dst
        if apply:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return "move", src, dst


def merge_dir(src_dir, dst_dir, apply, overwrite=False):
    actions = []
    if not src_dir.is_dir():
        return actions
    dst_dir.mkdir(parents=True, exist_ok=True) if apply else None
    for src in sorted(src_dir.iterdir()):
        dst = dst_dir / src.name
        actions.append(move_path(src, dst, apply, overwrite=overwrite))
    return actions


def cleanup_empty_dirs(root, apply):
    removed = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                if not any(path.iterdir()):
                    removed.append(path)
                    if apply:
                        path.rmdir()
            except OSError:
                pass
    return removed


def fix_scene(scene_dir, apply, overwrite):
    actions = []

    nested_root = scene_dir / "ground_truth"
    if not nested_root.is_dir():
        return actions

    actions.append(
        move_path(
            nested_root / "be_seq.csv",
            scene_dir / "be_seq.csv",
            apply,
            overwrite=overwrite,
        )
    )

    nested_camera = nested_root / "ground_truth" / "camera"
    expected_camera = scene_dir / "ground_truth" / "camera"
    if nested_camera.is_dir():
        if expected_camera.exists() and expected_camera.is_dir():
            actions.extend(merge_dir(nested_camera, expected_camera, apply, overwrite=overwrite))
        else:
            actions.append(move_path(nested_camera, expected_camera, apply, overwrite=overwrite))

    return [action for action in actions if action[0] != "missing"]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fix BEDLAM scene layout when be_seq.csv and ground_truth/camera are "
            "nested one extra level under scene/ground_truth."
        )
    )
    parser.add_argument("--root", type=str, default="/workspace/BEDLAM_Dataset")
    parser.add_argument("--apply", action="store_true", help="actually move files")
    parser.add_argument("--overwrite", action="store_true", help="replace existing destination files")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    all_actions = []
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        actions = fix_scene(scene_dir, args.apply, args.overwrite)
        for status, src, dst in actions:
            all_actions.append((status, src, dst))
            print("%s: %s -> %s" % (status, src, dst))

    removed = cleanup_empty_dirs(root, args.apply)
    for path in removed:
        print("rmdir: %s" % path)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print("%s: %d move/check actions, %d empty dirs" % (mode, len(all_actions), len(removed)))
    if not args.apply:
        print("Re-run with --apply to move files.")


if __name__ == "__main__":
    main()
