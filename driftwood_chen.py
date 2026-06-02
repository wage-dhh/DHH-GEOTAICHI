from geotaichi import *

init(device_memory_GB=15,debug=False)

dempm = DEMPM()
outputdata=r'E:\taichi_results\dirftwood_total\lq_woody'
dempm.set_configuration(domain=ti.Vector([131, 6, 22]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.5,
                      alphaPIC=0.005,
                      mapping="MUSL",
                      shape_function="GIMP",
material_type="Fluid",
#velocity_projection="Affine", # 'Taylor'->TPIC, 'Affine'->APIC, PIC/FLIP'
                     #stabilize="B-Bar Method",
                      gravity=ti.Vector([1.7, 0., -9.65]))

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([1.7, 0., -9.65]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         1e-4,
                      "SimulationTime":   60,
                      "SaveInterval":     0.5,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_particle_number": 10860,
                                "max_sphere_number": 194,
                                "max_clump_number": 65,
"max_patch_number": 115,
                                "compaction_ratio": [0.3,0.3],
"body_coordination_number": 162,
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           726000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0,
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    400,
                                  "wall_coordination_number":    200,
                                  "compaction_ratio": 0.3
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.25,
                                "TorqueLocalDamping": 0.7
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\impact_force_test\driftwood\driftwood_chen\after_rock_dwc.txt",
                               "GroupID": 0,
                               "MaterialID":2,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})
                               


dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\small-cistern\rock_dwc_two.txt",
                               "GroupID": 0,
                               "MaterialID":2,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Fix","Fix","Fix"],
                               "FixAngularVelocity": ["Fix","Fix","Fix"]
                               }})
'''

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
template_data = read_cylinder_template(r"E:\Taichi\small-cistern\driftwood_chen.txt")
dempm.dem.add_template(template=template_data)

'''
dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "Clump",
                     "Template":[
                                {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
"FixVelocity": ["Fix","Fix","Fix"],
"FixAngularVelocity": ["Fix","Fix","Fix"],
                                  "BodyPoint": [11.3, 2.5, 5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  }]})
'''

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "Clump",
                     "Template":[
                                {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.3, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },

                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.6, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.9, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.2, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.5, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.8, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.1, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.4, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.7, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.0, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.3, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.6, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.9, 3, 14.3],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  }
                      
                         ,
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.3, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.6, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.9, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.2, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.5, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.8, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.1, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.4, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.7, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.0, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.3, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.6, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.9, 3, 14.6],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.3, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.6, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.9, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.2, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.5, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.8, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.1, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.4, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.7, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.0, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.3, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.6, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.9, 3,14.9],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.3, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.6, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.9, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.2, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.5, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.8, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.1, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.4, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.7, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.0, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.3, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.6, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.9, 3,15.2],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.3, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.6, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [11.9, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },
                         {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.2, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.5, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [12.8, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.1, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.4, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [13.7, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.0, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.3, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.6, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  },{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [14.9, 3, 15.5],
                                  "Radius": 1.0*0.24666139428429784/1,
                                  "BodyOrientation": "constant"
                                  }
                                ]})
dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=0,
                       materialID2=0,
                       property={
                           "ShearModulus": 5e7,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.01
                       })

dempm.dem.add_property(materialID1=0,
                       materialID2=1,
                       property={
                           "ShearModulus": 5e7,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\cistern-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\pool-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\baffle-rbb.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\baffle-rbf.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\flume-top.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   5,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\temporary_woody.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   6,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\temporary_water.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   7,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\flume-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   8,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\small-cistern\flumeadd-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.select_save_data(clump=True,particle=True, wall=True)
'''
dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1000,
                               "YoungModulus":                  8e4,
                               "PoissionRatio":                 0.25,
                               "Friction":                      0,
                               "Dilation":                      0.0,
                               "Cohesion":                      0.0,
                               "Tensile":                       0.0
                 })

dempm.mpm.add_material(model="Bingham",
                 material={
                               "MaterialID":           1,
                               "Density":              1800.,
                               "Modulus":              1e6,
                               "Viscosity":            0.001,
                               "YieldStress":          1,
                               "CriticalStrainRate":   1,
                 })
'''
dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1800.,
                               "Modulus":              1e6,
                               "Viscosity":            0.001,
                 })
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.2,0.2,0.2])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 10]),
                            "BoundingBoxSize": ti.Vector([11, 6, 8]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])
dempm.mpm.add_region(region=[{
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([11.0, 1, 10]),
                            "BoundingBoxSize": ti.Vector([5, 4, 4]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
dempm.mpm.add_body(body={
    "Template": [{
        "RegionName": "region2",
        "nParticlesPerCell": 2,
        "BodyID": 0,
        "MaterialID": 1,
        "ParticleStress": {
            "GravityField": False,
            "InternalStress": ti.Vector([-0., -0., -0., 0., 0., 0.])
        },
        "InitialVelocity": ti.Vector([0, 0, 0]),
        "FixVelocity": ["Free", "Free", "Free"]

    }]
})
dempm.mpm.select_save_data()
dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Fluid Particle",
                           particle_wall_contact_model="Fluid Particle")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e7,
                                 "NormalViscousDamping":          0.2
                            }, dType='particle-particle')

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e8,
                                 "NormalViscousDamping":          0.6
                            }, dType='particle-wall')

dempm.run()
'''
dempm.dem.update_wall_status(wallID=6, property_name="Status", value=0)
dempm.dem.update_wall_status(wallID=4, property_name="Status", value=0)
dempm.dem.update_wall_status(wallID=5, property_name="Status", value=0)
dempm.mpm.modify_parameters(SimulationTime=6.0,background_damping=0.05)
dempm.modify_parameters(SimulationTime=6.0)
dempm.run()
dempm.modify_parameters(SimulationTime=12.0,SaveInterval=0.01)
dempm.run()

dempm.modify_parameters(SimulationTime=11.0,SaveInterval=0.03)
dempm.run()
'''

dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

