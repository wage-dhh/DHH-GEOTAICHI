from geotaichi import *

init(device_memory_GB=15,debug=True)

dempm = DEMPM()
outputdata=r'E:\taichi_results\clump-txt\clump-txt'
dempm.set_configuration(domain=ti.Vector([0.1, 0.1, 0.15]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.,
                      alphaPIC=0.0,
                      mapping="MUSL",
                      shape_function="GIMP",
material_type="Fluid",

                      gravity=ti.Vector([0, 0., -9.8]))

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([0, 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         1e-4,
                      "SimulationTime":   0.1,
                      "SaveInterval":     0.005,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_particle_number": 70,
                                "max_sphere_number": 7,
                                "max_clump_number": 12,
"max_patch_number": 20,
                                "compaction_ratio": 0.5,
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           1000752,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0,
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    26,
                                  "wall_coordination_number":    20,
                                  "compaction_ratio": 0.5
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            860,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\taichi2025.2.25\GeoTaichi-main2025.2.25\code\clump_woody\two_rock.txt",
                               "GroupID": 0,
                               "MaterialID":0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Clump",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "ClumpFile": r"E:\Taichi\taichi2025.2.25\GeoTaichi-main2025.2.25\example\dem\GranularPackings\clump\ClumpPacking-test.txt",
                               "PebbleFile": r"E:\Taichi\taichi2025.2.25\GeoTaichi-main2025.2.25\example\dem\GranularPackings\clump\PebblePacking-test.txt",
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})
'''
##########################################################################################################################

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
    template["NSphere"] = len(template["Pebble"])
    return template
template_data = read_cylinder_template(r"E:\Taichi\wall\small-cistern\clump.txt")
dempm.dem.add_template(template=template_data)
def read_clump_template(file_path,fixed_radius):
    template_list=[]
    with open(file_path,"r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        x, y, z = map(float, parts)
        entry = {
            "Name":"clump1",
            "GroupID":0,
            "MaterialID": 0,
            "BodyPoint":[x, y, z],
            "Radius": fixed_radius,
            "BodyOrientation": "constant",
        }
        template_list.append(entry)
    return template_list
fixed_radius = 1.0*0.5591253750588737/40
template_data = read_clump_template( r"E:\Taichi\taichi2025.2.25\GeoTaichi-main2025.2.25\code\clump_woody\position.txt", fixed_radius)
body_data = {
    "GenerateType":"Create",
    "BodyType":"Clump",
    "Template": template_data
}
dempm.dem.create_body(body=body_data)
'''
########################################################################################################
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
                           "ShearModulus": 5e8,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })


dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\taichi2025.2.25\GeoTaichi-main2025.2.25\code\clump_woody\flume-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.select_save_data(clump=True,particle=True, wall=True)

dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              5e6,
                               "Viscosity":            1e-3,
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.05, 0.05, 0.05])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.0]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.1]),
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

dempm.mpm.select_save_data()
dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Fluid Particle",
                           particle_wall_contact_model="Fluid Particle")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e3,
                                 "NormalViscousDamping":          0.,
'NonSlipFraction':0
                            }, dType='particle-particle')

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e3,
                                 "NormalViscousDamping":          0.,
'NonSlipFraction':0
                            }, dType='particle-wall')

dempm.run()

dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

