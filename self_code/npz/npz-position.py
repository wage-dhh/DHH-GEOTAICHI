'''
import os
import re
import numpy as np
import pandas as pd

# 文件夹路径
folder_path = r"E:\taichi_results\waterball\waterball0.001\particles"  # 替换为你的文件夹路径

# 定义保存文件夹和汇总文件路径
output_folder = r"E:\taichi\self_code\npz\water0.001_fx"
summary_file_path = os.path.join(output_folder, "water0.001_fx.xlsx")

# 如果保存文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)

# 定义文件名的正则表达式模式
file_pattern = r"DEMParticle\d{6}\.npz"  # 匹配以 DEMWall 后跟 6 位数字并以 .npz 结尾的文件名

# 获取所有符合条件的文件路径
npz_files = [f for f in os.listdir(folder_path) if re.match(file_pattern, f)]

# 准备存储所有 Z 值的列表
all_z_values = []

# 读取每个 npz 文件，提取 z 方向的值并进行调整
constant = 0.1  # 你希望减去的常数
for npz_file in npz_files:
    npz_file_path = os.path.join(folder_path, npz_file)

    # 加载 .npz 文件
    data = np.load(npz_file_path)

    # 提取 'position' 数据（假设 'position' 在 .npz 文件中是一个键）
    position = data['position']

    # 提取 z 方向的数据
    z_values = position[:, 2]  # 假设 z 值在第三列

    # 将 z 值减去常数
    z_values_adjusted = -z_values + constant

    # 将文件名和调整后的 z 值添加到结果列表中
    all_z_values.extend(zip([npz_file] * len(z_values_adjusted), z_values_adjusted))

# 将结果转换为 pandas DataFrame
df = pd.DataFrame(all_z_values, columns=['File Name', 'Adjusted Z Values'])

# 将结果保存为 Excel 文件
df.to_excel(summary_file_path, index=False)

print(f"已将调整后的 Z 值保存到 {summary_file_path}")

'''
import os
import re
import numpy as np
import pandas as pd

# 文件夹路径
folder_path = r"E:\taichi_results\dirftwood_total\clump-woody65——10\particles"  # 替换为你的文件夹路径

# 定义保存文件夹和汇总文件路径
output_folder = r"E:\taichi_results\data_analysis\water-position\clump-woody65——10_fx"
summary_file_path = os.path.join(output_folder, "clump-woody65——10_fx.xlsx")

# 如果保存文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)

# 定义文件名的正则表达式模式
file_pattern = r"MPMParticle\d{6}\.npz"  # 匹配以 DEMParticle 后跟 6 位数字并以 .npz 结尾的文件名

# 获取所有符合条件的文件路径
npz_files = [f for f in os.listdir(folder_path) if re.match(file_pattern, f)]

# 检查是否找到了符合条件的文件
if not npz_files:
    print("没有找到符合条件的文件。请检查文件路径和正则表达式。")
else:
    print(f"找到 {len(npz_files)} 个文件：{npz_files}")

# 设定常数
constant = 0. # 你希望减去的常数
a = 61  # 设定阈值 a，粒子数目大于该值的将被计算

# 准备存储粒子数目的列表
particle_counts = []  # 用于存储每个文件中 z_values_adjusted 大于 a 的粒子数目

# 读取每个 npz 文件，提取 z 方向的值并进行调整
for npz_file in npz_files:
    npz_file_path = os.path.join(folder_path, npz_file)

    # 加载 .npz 文件
    data = np.load(npz_file_path)

    # 提取 'position' 数据（假设 'position' 在 .npz 文件中是一个键）
    position = data['position']

    # 提取 z 方向的数据
    z_values = position[:, 0]  # 假设 z 值在第三列

    # 将 z 值减去常数
    z_values_adjusted = z_values + constant

    # 计算 z_values_adjusted 大于 a 的粒子数目
    count_greater_than_a = np.sum(z_values_adjusted > a)  # 计算大于 a 的粒子数目
    particle_counts.append((npz_file, count_greater_than_a))  # 将文件名和粒子数目添加到列表

# 检查是否有粒子数目数据
if particle_counts:
    print(f"粒子数目数据：{particle_counts}")
else:
    print("没有计算粒子数目。请检查 z 值调整的结果是否符合预期。")

# 将粒子数目保存到 pandas DataFrame
df_particle_counts = pd.DataFrame(particle_counts, columns=['File Name', 'Particles Greater Than A'])

# 检查数据框是否正确
print(f"Particle Counts 数据：\n{df_particle_counts.head()}")

# 将粒子数目保存到 Excel 文件
df_particle_counts.to_excel(summary_file_path, index=False, sheet_name='Particle Counts')

print(f"已将粒子数目保存到 {summary_file_path}")
