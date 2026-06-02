from geotaichi import *

init(device_memory_GB=12,debug=False, gpu_id=0)
#init(arch="cpu", debug=True)

dempm = DEMPM()
outputdata=r'F:\taichi_results\DHH\sim01_rock10'
dempm.set_configuration(domain=ti.Vector([100, 30, 30]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.01,
                      alphaPIC=0.01,
                      mapping="MUSL",
                      shape_function="QuadBSpline",
#shape_function="GIMP",
material_type="Fluid",
#velocity_projection="Affine", # 'Taylor'->TPIC, 'Affine'->APIC, PIC/FLIP'
                     #stabilize="B-Bar Method",
                      gravity=ti.Vector([4.9, 0., -8.66]))

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([4.9, 0., -8.66]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         1e-4,
                      "SimulationTime":   1,
                      "CFL": 0.9,
                      "SaveInterval":     0.1,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 1750,
                                "max_sphere_number": 1750,
                                #"max_clump_number": 80,
                                "body_coordination_number": 40,
                                "wall_coordination_number": 58,
                                "max_patch_number": 58,
                                "wall_per_cell": 10,
                                "compaction_ratio": [0.8, 0.8],
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           470000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0,
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    130,
                                  "wall_coordination_number":    58,
                                  "compaction_ratio": 0.8
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })


dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })

dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([1, 2, 2.5]),
                               "FixVelocity": ["Fix","Fix","Fix"],
                               "FixAngularVelocity": ["Fix","Fix","Fix"],
                               "Radius": 0.1,
                               "BodyOrientation": "uniform"}]})
'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"F:\dhh\Coordinates_and_radius\015_10_reordered.txt",
                               "GroupID": 0,
                               "MaterialID": 1,
                               "InitialVelocity": ti.Vector([0., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "FixVelocity": ["Free", "Free", "Free"],
                               "FixAngularVelocity": ["Free", "Free", "Free"]
                               }})
'''
'''
dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 1,
                               "InitialVelocity": ti.Vector([2, 0., 0]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               #"BodyPoint": ti.Vector([0.1, 0.11, 0.15]),
"BodyPoint": ti.Vector([0.56, 0.11, 0.07]),
                               "FixVelocity": ["Free", "Free", "Free"],
                               "FixAngularVelocity": ["Free", "Free", "Free"],
                               "Radius": 0.01,
                               "BodyOrientation": "uniform"}]})
'''
dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=1,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e8,
                           "Possion": 0.3,
                           "Friction": 0.5,
                           "Restitution": 0.5
                       })

dempm.dem.add_property(materialID1=1,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e8,
                           "Possion": 0.3,
                           "Friction": 0.,
                           "Restitution": 0.5
                       })


dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"F:\dhh\flume\back.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"F:\dhh\flume\bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"F:\dhh\flume\front.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"F:\dhh\flume\left.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   5,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"F:\dhh\flume\right.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.select_save_data(clump=True,particle=True, wall=True,sphere=True)

dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1500.,
                               "Modulus":              1e6,
                               "Viscosity":            1,
                 })
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })

dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.32, 0.1]),
#"BoundingBoxSize": ti.Vector([0.225, 0.32, 0.10]),
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
                                       "InitialVelocity": ti.Vector([0, 0, 0]),
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
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
                                 'NonSlipFraction': 0.0
                            }, dType='particle-wall')

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
                                 'NonSlipFraction': 0
                            }, dType='particle-particle')

dempm.run()
'''
dempm.dem.update_contact_properties(materialID1=3, materialID2=1, property_name="ShearModulus",value=0)
dempm.modify_parameters(SimulationTime=8)
dempm.run()
'''
dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

