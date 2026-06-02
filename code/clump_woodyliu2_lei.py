from geotaichi import *

init(device_memory_GB=45,debug=True, gpu_id=1)

dempm = DEMPM()
outputdata=r'E:\taichi_results\radius\radius0.04\lei'
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
                      gravity=ti.Vector([1.7, 0., -9.65]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         2e-4,
                      "SimulationTime":   1.5,
                      "CFL": 0.9,
                      "SaveInterval":     0.01,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 5,
                                "max_particle_number": 1594,
                                "max_sphere_number": 139,
                                "max_clump_number": 50,
                                #"body_coordination_number":130,
                                "max_patch_number": 38,
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
                                  "body_coordination_number":    24,
                                  "wall_coordination_number":    38,
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

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\rock\rocklei_after.txt",
                               "GroupID": 0,
                               "MaterialID":4,
                               "InitialVelocity": ti.Vector([8.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free", "Free", "Free"],
                               "FixAngularVelocity": ["Free", "Free", "Free"]
                               }})

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Clump",
                               "GroupID": 0,
                               "MaterialID": 3,
                               "ClumpFile": r"E:\Taichi\clump\radiuslei\clump.txt",
                               "PebbleFile": r"E:\Taichi\clump\radiuslei\pebble.txt",
                               "InitialVelocity": ti.Vector([8,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})
'''

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "NPZ",
                   "Template": {"Restart": True,
                                "BodyType": "Clump",
                                "ParticleFile": r"E:\Taichi\clump\gra_radius\radius0.04\radiuslei\particle.npz",
                                "SphereFile": r"E:\Taichi\clump\gra_radius\radius0.04\radiuslei\sphere.npz",
                                "ClumpFile": r"E:\Taichi\clump\gra_radius\radius0.04\radiuslei\clump.npz"
                               }})
'''
dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=3,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.3,
                           "Restitution": 0.0
                       })

dempm.dem.add_property(materialID1=3,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })

dempm.dem.add_property(materialID1=3,
                       materialID2=2,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.3,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=3,
                       materialID2=3,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=4,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.3,
                           "Restitution": 0.0
                       })

dempm.dem.add_property(materialID1=4,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })

dempm.dem.add_property(materialID1=4,
                       materialID2=2,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.3,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=4,
                       materialID2=3,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=4,
                       materialID2=4,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\left_front_behind.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-right.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   2,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\baffle-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-top.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.select_save_data(clump=True,particle=True, wall=True,sphere=True)
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
                               "Density":              1200.,
                               "Modulus":              1e6,
                               "Viscosity":            1,
                 })
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })
'''
dempm.mpm.add_body_from_file(body={
                                    "FileType": "TXT",
                                    "Template":  {
                                                   "BodyID": 0,
                                                   "MaterialID": 1,
                                                   "ParticleFile": r"E:\taichi_result\radius0.04\radius0.04\particles\MPMParticle000080.txt"
                                                  }
                   })

'''
dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0, 0]),
                            "BoundingBoxSize": ti.Vector([3, 0.6, 0.2]),
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
                                       "InitialVelocity":ti.Vector([8, 0, 0]),
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
'NonSlipFraction':0.001
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

