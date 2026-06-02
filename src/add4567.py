import pandas as pd

# ====== 1. 输入文件路径 ======
input_file = "F:\txmpm\dhhmpm\cobe_1P.txt"   # 改成你的文件名
output_file = "F:\txmpm\dhhmpm\addcobe_1P.txt"

# 如果你的数据是空格或多个空格分隔，用 sep='\s+'
df = pd.read_csv(input_file, sep='\s+', header=None)

# ====== 2. 修改第4列（索引3）=====
df.iloc[:, 3] = df.iloc[:, 3] + 0.00000125

# ====== 3. 修改第5~7列（索引4:7）=====
df.iloc[:, 4:7] = df.iloc[:, 4:7] + 0.025

# ====== 4. 保存到新文件 ======
df.to_csv(output_file, sep=' ', index=False, header=False)

print(f"处理完成，输出文件：{output_file}")
