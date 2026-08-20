import os
import glob
import re
import argparse
import pandas as pd
import json

def extract_timestamp(path):
    nums = ''.join(filter(str.isdigit, os.path.basename(path)))
    return int(nums) if nums else 0

def extract_frame_id(path):
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else -1

def build_one_episode(dataset_root, gt_root,result_root, base_seq, condition):
    seq_name = base_seq + condition
    rgb_dir = os.path.join(dataset_root, seq_name, "rgb")
    depth_dir = os.path.join(dataset_root, seq_name, "depth")
    gt_dir = os.path.join(gt_root, base_seq, "annotated_poses")
    pred_dir = os.path.join(result_root, base_seq, seq_name)

    assert os.path.exists(rgb_dir), rgb_dir
    assert os.path.exists(depth_dir), depth_dir
    assert os.path.exists(gt_dir), gt_dir
    assert os.path.exists(pred_dir), pred_dir

    gt_dict = {extract_frame_id(f): f for f in glob.glob(os.path.join(gt_dir, "*.txt"))}
    pred_dict = {extract_frame_id(f): f for f in glob.glob(os.path.join(pred_dir, "*.txt"))}

    assert set(gt_dict) == set(pred_dict), f"{seq_name}: GT/Pred frame mismatch"

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")), key=extract_timestamp)
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")), key=extract_timestamp)
    frame_ids = sorted(gt_dict.keys())

    assert len(rgb_files) == len(frame_ids), f"{seq_name}: RGB count mismatch"
    assert len(depth_files) == len(frame_ids), f"{seq_name}: Depth count mismatch"

    rows = []
    for i, fid in enumerate(frame_ids):
        rows.append({
            "base_sequence": base_seq,
            "condition": condition.strip("_"),
            "sequence": seq_name,
            "frame_id": fid,
            "rgb_name": os.path.basename(rgb_files[i]),
            "depth_name": os.path.basename(depth_files[i]),
            "gt_name": os.path.basename(gt_dict[fid]),
            "pred_name": os.path.basename(pred_dict[fid]),
            "rgb_path": os.path.relpath(rgb_files[i], dataset_root),
            "depth_path": os.path.relpath(depth_files[i], dataset_root),
            "gt_path": os.path.relpath(gt_dict[fid], dataset_root),
            "pred_path": os.path.relpath(pred_dict[fid], result_root)
        })
    return rows

def build_all_manifest(dataset_root, gt_root, result_root, config_path):

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_sequences = config["base_sequences"]
    common_conditions = config["common_conditions"]
    extra_conditions = config["extra_conditions"]

    rows = []
    for seq in base_sequences:
        for cond in common_conditions:
            rows += build_one_episode(dataset_root,gt_root, result_root, seq, cond)
        for cond in extra_conditions.get(seq, []):
            rows += build_one_episode(dataset_root, gt_root,result_root, seq, cond)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--gt_root", required=True)
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="dataset_manifest_all.csv")
    args = parser.parse_args()

    df = build_all_manifest(args.dataset_root,args.gt_root, args.result_root,args.config)
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(f"Frames: {len(df)}, Episodes: {df.sequence.nunique()}")
    print(df.head())
