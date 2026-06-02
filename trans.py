import numpy as np

file_path = r'F:\taichi_results\DHH\EXAMPLE18_30_12\particlestxt\MPMParticle000030.txt'

col1 = []

with open(file_path, 'r') as f:
    for line in f:
        line = line.strip()

        # 跳过无效行
        if not line or '===' in line:
            continue

        parts = line.split()

        try:
            col1.append(float(parts[0]))
        except:
            continue

col1 = np.array(col1)

ratio = np.sum(col1 > 2.65) / len(col1)

print("总数:", len(col1))
print("占比:", ratio)