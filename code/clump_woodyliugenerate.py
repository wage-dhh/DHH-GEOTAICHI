from geotaichi import *

init(device_memory_GB=15,debug=False,gpu_id=1)

dempm = DEMPM()
outputdata=r'E:\taichi_results\radius\radius0.4_4_generate11'
dempm.set_configuration(domain=ti.Vector([7, 0.6, 0.6]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.01,
                      alphaPIC=0.01,
                      mapping="MUSL",
                      shape_function="QuadBSpline",
material_type="Fluid",
#velocity_projection="Affine", # 'Taylor'->TPIC, 'Affine'->APIC, PIC/FLIP'
                     #stabilize="B-Bar Method",
                      gravity=ti.Vector([1.7, 0., -9.65]))

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([0, 0., -9.81]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         5e-4,
                      "SimulationTime":   0.5,
                      "CFL": 0.85,
                      "SaveInterval":     0.01,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 5,
                                "max_particle_number": 2194,
                                "max_sphere_number": 196,
                                "max_clump_number": 100,
                                "max_patch_number": 34,
                                "compaction_ratio": [0.4,0.4],
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           4560000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0,
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    17,
                                  "wall_coordination_number":    34,
                                  "compaction_ratio": 0.4
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })
dempm.dem.add_attribute(materialID=2,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })
dempm.dem.add_attribute(materialID=3,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })
dempm.dem.add_attribute(materialID=4,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })
'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\rock\rock004_15after.txt",
                               "GroupID": 0,
                               "MaterialID":2,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Fix", "Fix", "Fix"],
                               "FixAngularVelocity": ["Fix", "Fix", "Fix"]
                               }})


dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 2,
                               "InitialVelocity": ti.Vector([0., 0., 0]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([4.7, 0.3, 0.05]),
                               "FixVelocity": ["Fix", "Fix", "Fix"],
                               "FixAngularVelocity": ["Fix", "Fix", "Fix"],
                               "Radius": 0.03,
                               "BodyOrientation": "uniform"}]})
'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Clump",
                               "GroupID": 1,
                               "MaterialID": 1,
                               "ClumpFile": r"E:\Taichi\clump\radius0.02_4.4_80\clump.txt",
                               "PebbleFile":r"E:\Taichi\clump\radius0.02_4.4_80\pebble.txt",
                               "InitialVelocity": ti.Vector([0,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})

dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=1,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.3,
                           "Restitution": 0.0
                       })

dempm.dem.add_property(materialID1=1,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })


'''
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-left32.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-front.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-behind.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-right52.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.select_save_data(clump=True,particle=True, wall=True,sphere=True)

dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1200.,
                               "Modulus":              1e6,
                               "Viscosity":            10,
                 })
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.05, 0.05, 0.05])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0, 0, 0]),
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
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]
                                       
                                   }]
                   })
dempm.mpm.select_save_data()
dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Fluid Particle",
                           particle_wall_contact_model="Fluid Particle")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e6,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':0
                            }, dType='particle-wall')

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e6,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':1
                            }, dType='particle-wall')
dempm.add_property(DEMmaterial=2,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e6,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':0.01
                            }, dType='particle-wall')
dempm.add_property(DEMmaterial=3,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e6,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':1
                            }, dType='particle-particle')
dempm.add_property(DEMmaterial=4,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e6,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':1
                            }, dType='particle-particle')

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

