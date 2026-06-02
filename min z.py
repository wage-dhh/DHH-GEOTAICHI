def find_min_z(file_path):
    min_z = float('inf')
    min_point = None
    in_position = False

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # 进入 position 区域
            if line == "=== position ===":
                in_position = True
                continue

            # 遇到下一个 === 就说明结束
            if in_position and line.startswith("==="):
                break

            # 解析数据
            if in_position and line:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = map(float, parts[:3])
                        if z < min_z:
                            min_z = z
                            min_point = (x, y, z)
                    except:
                        continue

    return min_point


# 使用
file_path = r"F:\taichi_results\DHH\EXAMPLE18_30_14\particlestxt\DEMParticle000030.txt" # 或你的本地路径
result = find_min_z(file_path)

print("最小Z及对应坐标:", result)