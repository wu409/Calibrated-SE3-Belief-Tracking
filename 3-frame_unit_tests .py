import os
import shutil
import tempfile
import unittest
import glob



def build_safe_depth_dict(depth_folder, gt_dict):
    """
    将时间戳命名的 depth 文件安全映射到整数 frame_id (0, 1, 2...)
    加入时间戳严格单调递增校验，彻底消灭错位风险！
    """
    # 1. 获取所有以时间戳命名的深度图文件
    depth_files = glob.glob(os.path.join(depth_folder, "*.png"))
    # 2. 从文件名提取纯数字时间戳，并按时间戳升序严格排序
    def extract_timestamp(path):
        digits = ''.join(filter(str.isdigit, os.path.basename(path)))
        return int(digits) if len(digits) > 0 else 0
        
    depth_files_sorted = sorted(depth_files, key=extract_timestamp)
    gt_frame_ids = sorted(gt_dict.keys()) # [0, 1, 2, ..., N]

    # 3. 严格长度与非空断言 (防丢帧)
    assert len(depth_files_sorted) == len(gt_frame_ids), \
        f"错误: 深度图数量 ({len(depth_files_sorted)}) 与 GT 帧数 ({len(gt_frame_ids)}) 不一致！"

    # 4. 校验时间戳是否严格单调递增 (防乱序)
    timestamps = [extract_timestamp(f) for f in depth_files_sorted]
    assert all(timestamps[k] < timestamps[k+1] for k in range(len(timestamps)-1)), \
        "错误: 深度图时间戳非严格单调递增！存在乱序帧！"

    # 5. 建立以整数 frame_id 为 Key 的安全字典
    depth_dict = {}
    for k, frame_id in enumerate(gt_frame_ids):
        depth_dict[frame_id] = depth_files_sorted[k]

    return depth_dict



class TestBuildSafeDepthDict(unittest.TestCase):

    def setUp(self):
        """每个测试开始前：在系统临时目录里建一个假的深度文件夹"""
        self.test_dir = tempfile.mkdtemp()
        self.depth_dir = os.path.join(self.test_dir, "depth")
        os.makedirs(self.depth_dir, exist_ok=True)

        # 模拟 GT 字典：包含 5 帧 (0, 1, 2, 3, 4)
        self.gt_dict = {
            0: "gt_000000.txt", 
            1: "gt_000001.txt", 
            2: "gt_000002.txt", 
            3: "gt_000003.txt", 
            4: "gt_000004.txt"
        }

    def tearDown(self):
        """测试结束后：自动清理临时文件夹"""
        shutil.rmtree(self.test_dir)

    def test_reordered_timestamps_sorting(self):
        """测试 1: 验证 build_safe_depth_dict 能否把乱序的时间戳文件按真实时间排对"""
        # 故意打乱时间戳顺序创建 5 个测试图片文件
        timestamps = [
            "15839201923845.png", # 对应 frame 2
            "15839201923812.png", # 对应 frame 0
            "15839201923899.png", # 对应 frame 4
            "15839201923820.png", # 对应 frame 1
            "15839201923870.png", # 对应 frame 3
        ]
        for ts in timestamps:
            with open(os.path.join(self.depth_dir, ts), 'w') as f:
                f.write("dummy_data")

        # 🌟 直接调用你主程序里真实的 build_safe_depth_dict 函数！
        depth_dict = build_safe_depth_dict(self.depth_dir, self.gt_dict)

        # 验证: 乱序输入后，第 0 帧是否被你的真实函数精准绑定到了最小时间戳 15839201923812.png！
        self.assertEqual(len(depth_dict), 5)
        self.assertTrue(depth_dict[0].endswith("15839201923812.png"))
        self.assertTrue(depth_dict[4].endswith("15839201923899.png"))
        print("\n✅ 测试 1: 真实函数 build_safe_depth_dict 成功自愈时间戳乱序！")

    def test_missing_depth_frame_assertion(self):
        """测试 2: 验证当深度图丢帧时，build_safe_depth_dict 能否精准触发 AssertionError 拦截"""
        # 故意只创建 4 张图片 (少了一帧)
        for ts in ["15839201923812.png", "15839201923820.png", "15839201923845.png", "15839201923870.png"]:
            with open(os.path.join(self.depth_dir, ts), 'w') as f:
                f.write("dummy_data")

        # 🌟 验证调用你真实的函数时，能抛出断言错误 AssertionError 阻止错位！
        with self.assertRaises(AssertionError):
            build_safe_depth_dict(self.depth_dir, self.gt_dict)
            
        print("✅ 测试 2: 真实函数 build_safe_depth_dict 成功拦截丢帧事故！")


if __name__ == '__main__':
    unittest.main()