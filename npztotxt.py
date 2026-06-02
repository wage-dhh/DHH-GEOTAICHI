import os
import numpy as np

# -------------------------
# 配置路径
# -------------------------
npz_folder = r'F:\taichi_results\DHH\EXAMPLE18_25_10\particles'  # npz 文件夹
txt_folder = r'F:\taichi_results\DHH\EXAMPLE18_25_10\particlestxt'  # 输出 txt 文件夹

os.makedirs(txt_folder, exist_ok=True)


# -------------------------
# 写入函数，兼容任意维度和空数组
# -------------------------
def write_array(f, arr):
    if np.size(arr) == 0:
        f.write('[Empty Array]\n')
        return

    arr = np.atleast_1d(arr)  # 保证至少 1D
    if arr.ndim == 1:
        f.write(' '.join(map(str, arr)) + '\n')
    else:
        # 高维数组展平每一行，逐行写入
        arr_2d = arr.reshape(arr.shape[0], -1)
        for row in arr_2d:
            f.write(' '.join(map(str, row)) + '\n')


# -------------------------
# 遍历 npz 文件
# -------------------------
for filename in os.listdir(npz_folder):
    if filename.endswith('.npz'):
        npz_path = os.path.join(npz_folder, filename)
        data = np.load(npz_path)

        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(txt_folder, txt_filename)

        with open(txt_path, 'w') as f:
            for key in data.files:
                f.write(f'=== {key} ===\n')
                arr = data[key]
                write_array(f, arr)
                f.write('\n')

        print(f'{filename} 已转换为 {txt_filename}')