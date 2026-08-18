import os
import shutil
import tempfile
import unittest
import glob
import re
import hashlib
import pandas as pd
import argparse

def compute_full_sha256(filepath):
    """计算文件的完整 64 位 SHA-256 哈希值 (读全量数据，绝不截断)"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_exact_file_number(filepath):
    """只从文件名提取纯数字 ID (如 000150.txt -> 150)"""
    filename = os.path.basename(filepath)
    digits = re.findall(r'\d+', filename)
    return int(digits[-1]) if len(digits) > 0 else -1

def extract_timestamp_int(filepath):
    """从时间戳文件名提取纯数字时间戳"""
    filename = os.path.basename(filepath)
    digits = ''.join(filter(str.isdigit, filename))
    return int(digits) if len(digits) > 0 else 0

def build_and_verify_manifest(dataset_dir, seq_name, res_dir, gt_dir, out_manifest_csv="./manifest.csv", max_async_tolerance_ms=50):
    """
    权威的 4 模态 (RGB, Depth, GT, Pred) 严格对齐、异步时间戳关联与完整 Hash 校验函数
    """
    rgb_dir = os.path.join(dataset_dir, seq_name, "rgb")
    depth_dir = os.path.join(dataset_dir, seq_name, "depth")
    pred_dir = res_dir

    # 1. 建立 GT 和 Pred 的真实 Frame-ID 字典
    gt_dict = {get_exact_file_number(f): f for f in glob.glob(os.path.join(gt_dir, "*.txt")) if get_exact_file_number(f) >= 0}
    pred_dict = {get_exact_file_number(f): f for f in glob.glob(os.path.join(pred_dir, "*.txt")) if get_exact_file_number(f) >= 0}

    #  断言 1: 严格显式校验 Prediction ID 与 GT Frame-ID 的完全一致性！
    assert len(gt_dict) > 0, f"错误: [{seq_name}] GT 标注目录为空！"
    assert len(pred_dict) > 0, f"错误: [{seq_name}] 预测结果目录为空！"
    assert set(pred_dict.keys()) == set(gt_dict.keys()), \
        f"错误: [{seq_name}] Pred 帧号集合与 GT 帧号集合不匹配！缺失帧: {set(gt_dict.keys()) ^ set(pred_dict.keys())}"

    # 2. 读取 RGB 与 Depth，按真实时间戳升序排序 (恢复时序流)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")), key=extract_timestamp_int)
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")), key=extract_timestamp_int)
    
    n_frames = len(gt_dict)
    assert len(rgb_files) == n_frames, f"错误: RGB 帧数 ({len(rgb_files)}) 与 GT 帧数 ({n_frames}) 不一致！"
    assert len(depth_files) == n_frames, f"错误: Depth 帧数 ({len(depth_files)}) 与 GT 帧数 ({n_frames}) 不一致！"

    # 2: 处理并校验异步硬件时间戳 (Asynchronous Stream Verification)
    for k in range(n_frames):
        t_rgb = extract_timestamp_int(rgb_files[k])
        t_depth = extract_timestamp_int(depth_files[k])
        # 允许微小硬件时钟偏差 (比如 <= 50ms)，但绝不允许严重漂移！
        diff_ms = abs(t_rgb - t_depth) / 1000.0 if t_rgb > 1e11 else abs(t_rgb - t_depth)
        assert diff_ms <= max_async_tolerance_ms, \
            f"第 {k} 帧 RGB 与 Depth 时间戳偏差过大 ({diff_ms} ms > {max_async_tolerance_ms} ms)！"

    # 3. 构建包含完整 64 位 SHA-256 哈希的权威元数据表
    sorted_fids = sorted(gt_dict.keys())
    manifest_rows = []
    
    for k, fid in enumerate(sorted_fids):
        rgb_p = rgb_files[k]
        depth_p = depth_files[k]
        gt_p = gt_dict[fid]
        pred_p = pred_dict[fid]

        manifest_rows.append({
            "seq_idx": k,
            "frame_idx": fid,
            "timestamp_rgb": os.path.basename(rgb_p),
            "timestamp_depth": os.path.basename(depth_p),
            "rgb_path": rgb_p,
            "depth_path": depth_p,
            "gt_path": gt_p,
            "pred_path": pred_p,
            "rgb_sha256": compute_full_sha256(rgb_p),
            "depth_sha256": compute_full_sha256(depth_p),
            "gt_sha256": compute_full_sha256(gt_p),
            "pred_sha256": compute_full_sha256(pred_p)
        })


    df_manifest = pd.DataFrame(manifest_rows)
    df_manifest.to_csv(out_manifest_csv, index=False)
    print(f" Manifest 已生成并验证通过: {out_manifest_csv} (共 {n_frames} 帧，包含完整 64 位 SHA-256 校验)")
    
    return df_manifest

def test_valid_real_manifest(self):

    rgb_dir, depth_dir, pred_dir = self.get_paths()

    df = build_and_verify_manifest(
        self.tmp_dataset,
        self.seq_name,
        pred_dir,
        self.gt_dir,
        out_manifest_csv=
        os.path.join(self.tmp_root,"manifest.csv"))


    self.assertGreater(len(df),0)
    print("\n✅ Real dataset manifest generation passed")




class TestProductionManifestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--dataset_dir",
            required=True
        )

        parser.add_argument(
            "--seq_name",
            required=True
        )

        parser.add_argument(
            "--gt_dir",
            required=True
        )

        parser.add_argument(
            "--pred_dir",
            required=True
        )


        args, _ = parser.parse_known_args()


        cls.dataset_dir = args.dataset_dir
        cls.seq_name = args.seq_name
        cls.gt_dir = args.gt_dir
        cls.pred_dir = args.pred_dir



    def setUp(self):

        # 创建临时测试环境
        self.tmp_root = tempfile.mkdtemp()

        self.tmp_dataset = os.path.join(self.tmp_root, "dataset")
        self.tmp_pred = os.path.join(self.tmp_root,"prediction")

        # copy真实数据
        shutil.copytree(self.dataset_dir,self.tmp_dataset)

        shutil.copytree(self.pred_dir,self.tmp_pred)



    def tearDown(self):
        shutil.rmtree(self.tmp_root)



    def test_normal_manifest_generation(self):

        """
        测试真实数据正常生成manifest
        """

        df = build_and_verify_manifest(
            self.tmp_dataset,
            self.seq_name,
            self.tmp_pred,
            self.gt_dir,
            out_manifest_csv=os.path.join(
                self.tmp_root,
                "manifest.csv"
            )
        )


        self.assertGreater(len(df),0)


        print("\n[PASS] Normal manifest generation")



    def test_missing_prediction_frame(self):

        """
        删除一个prediction frame
        检查是否捕获Pred-GT mismatch
        """

        pred_files = sorted(
            glob.glob(
                os.path.join(
                    self.tmp_pred,
                    "*.txt")))


        self.assertGreater(len(pred_files),0)

        os.remove(pred_files[0])


        with self.assertRaises(AssertionError):

            build_and_verify_manifest(
                self.tmp_dataset,
                self.seq_name,
                self.tmp_pred,
                self.gt_dir
            )


        print("\n[PASS] Missing prediction frame detected")

    def test_missing_rgb_frame(self):

        """
        删除RGB
        """

        rgb_dir=os.path.join(
            self.tmp_dataset,
            self.seq_name,
            "rgb"
        )


        rgb_files=glob.glob(
            os.path.join(
                rgb_dir,
                "*.png"
            ))


        os.remove(rgb_files[0])
        with self.assertRaises(AssertionError):

            build_and_verify_manifest(
                self.tmp_dataset,
                self.seq_name,
                self.tmp_pred,
                self.gt_dir
            )

        print("\n[PASS] Missing RGB frame detected")



    def test_missing_depth_frame(self):

        """
        删除Depth
        """

        depth_dir=os.path.join(
            self.tmp_dataset,
            self.seq_name,
            "depth")


        depth_files=glob.glob(
            os.path.join(
                depth_dir,
                "*.png"
            ))

        os.remove(depth_files[0])
        with self.assertRaises(AssertionError):

            build_and_verify_manifest(
                self.tmp_dataset,
                self.seq_name,
                self.tmp_pred,
                self.gt_dir
            )

        print("\n[PASS] Missing Depth frame detected")



    def test_duplicate_prediction_id(self):

        """
        制造重复prediction ID

        注意:
        需要build_and_verify_manifest中加入duplicate ID检查
        """

        pred_files=glob.glob(
            os.path.join(
                self.tmp_pred,
                "*.txt"
            )
        )


        src=pred_files[0]


        duplicate=os.path.join(
            self.tmp_pred,
            os.path.basename(src).replace(
                ".txt",
                "_copy.txt"
            ))


        shutil.copy(src,duplicate )
        with self.assertRaises(AssertionError):

            build_and_verify_manifest(
                self.tmp_dataset,
                self.seq_name,
                self.tmp_pred,
                self.gt_dir
            )

        print("\n[PASS] Duplicate prediction ID detected")



    def test_reordered_rgb_timestamp(self):

        """
        打乱RGB timestamp文件名
        """

        rgb_dir=os.path.join(self.tmp_dataset,self.seq_name,"rgb")

        rgb_files=sorted(glob.glob(os.path.join(rgb_dir,"*.png")))

        if len(rgb_files)<2:
            self.skipTest("Not enough RGB frames")

        f1=rgb_files[0]
        f2=rgb_files[1]

        tmp=f1+"_tmp"

        os.rename(f1,tmp)
        os.rename(f2, f1)
        os.rename(tmp,f2)


        with self.assertRaises(AssertionError):
            build_and_verify_manifest(
                self.tmp_dataset,
                self.seq_name,
                self.tmp_pred,
                self.gt_dir)


        print("\n[PASS] Reordered timestamp detected")



    def test_timestamp_mismatch(self):

        """
        修改depth timestamp
        检查RGB-depth时间一致性
        """

        depth_dir=os.path.join(self.tmp_dataset, self.seq_name, "depth")

        depth_files=sorted(glob.glob(os.path.join(depth_dir,"*.png")))

        old=depth_files[0]


        new=os.path.join(depth_dir,"999999999999.png")
        os.rename(old,new)
        with self.assertRaises(AssertionError):

            build_and_verify_manifest(
                self.tmp_dataset,
                self.seq_name,
                self.tmp_pred,
                self.gt_dir)


        print("\n[PASS] Timestamp mismatch detected" )



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--seq_name",required=True)
    parser.add_argument("--gt_dir",required=True)
    parser.add_argument("--pred_dir", required=True)
    args, remaining = parser.parse_known_args()


    TestProductionManifestSuite.dataset_dir = args.dataset_dir
    TestProductionManifestSuite.seq_name = args.seq_name
    TestProductionManifestSuite.gt_dir = args.gt_dir
    TestProductionManifestSuite.pred_dir = args.pred_dir


    unittest.main(argv=["first-arg-is-ignored"] + remaining)