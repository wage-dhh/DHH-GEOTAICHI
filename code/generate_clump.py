'''
dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 2,
                                 "Pebble": [{
                                             "Position": ti.Vector([-0.5, 0., 0.]),
                                             "Radius": 1.
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 0.]),
                                             "Radius": 1.
                                            }]
                                 })

def read_cylinder_template(file_path):
    template = {"Pebble": []}
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        x, y, z, radius = map(float, parts)
        template["Pebble"].append({
            "Position": ti.Vector([x, y, z]),
            "Radius": radius
        })
    template["Name"] = "clump1"
    template["NSphere"] = len(template["Pebble"])  # 计算小球数量
    return template
template_data = read_cylinder_template(r"E:\Taichi\wall\small-cistern\clump.txt")
dem.add_template(template=template_data)
'''