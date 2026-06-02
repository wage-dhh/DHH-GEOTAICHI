import os
import re
import numpy as np
import pandas as pd

def process_npz_folder(
    folder_path: str,
    output_excel: str,
    x_threshold: float = 1.0
):
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)

    file_pattern = re.compile(r"DEMParticle(\d{6})\.npz")
    #file_pattern = re.compile(r"DEMClump(\d{6})\.npz")

    # 先按编号排序，保证“前一个”“后一个”有明确顺序
    files = sorted(
        [f for f in os.listdir(folder_path) if file_pattern.match(f)],
        key=lambda f: int(file_pattern.match(f).group(1))
    )

    records = []
    prev_count = None   # 保存前一个文件的 Count

    for fname in files:
        data = np.load(os.path.join(folder_path, fname))
        #x_all = data["position"][:, 0]
        x_all = data["centerOfMass"][:, 0]

        # 统计 x > threshold 的粒子数
        count_gt = int((x_all > x_threshold).sum())

        # 与前一个文件做差
        if prev_count is None:
            dcount = ""          # 首个文件无前驱，用空串占位
        else:
            dcount = count_gt - prev_count

        # 记录
        '''
        records.append({
            "File Name": fname,
            f"Count (x > {x_threshold})": count_gt,
            "ΔCount vs. prev": dcount
        })
        '''
        records.append({
            "File Name": fname,
            f"Count (x > {x_threshold})": count_gt,
        })

        prev_count = count_gt

    pd.DataFrame.from_records(records).to_excel(
        output_excel, index=False, sheet_name="Count_Stats")
    print(f"✓ 统计完成，结果已保存到: {output_excel}")


# ---------------- 示例运行 ----------------
if __name__ == "__main__":
    folder_path = r"H:\taichi_results\radius\radius0.04\radius_small42fT\particles"
    output_excel = r"E:\taichi_result\radius\radius0.04\6_17\radius_small42fT.xlsx"
    process_npz_folder(folder_path, output_excel, x_threshold=5.3)