from geotaichi import *

init(device_memory_GB=45,debug=False, gpu_id=1)

dempm = DEMPM()
outputdata=r'H:\taichi_results\radius\4times_sim\simulation4420_1'
dempm.set_configuration(domain=ti.Vector([7, 0.6, 2]),
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
                      "SimulationTime":   4,
                      "CFL": 0.9,
                      "SaveInterval":     0.02,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 4,
                                "max_particle_number": 2000,
                                "max_sphere_number": 1,
                                "max_clump_number": 80,
                                "body_coordination_number": 40,
                                "wall_coordination_number": 42,
                                "max_patch_number": 42,
                                "wall_per_cell": 4,
                                "compaction_ratio": [0.5, 0.5],
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           4500000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0,
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    130,
                                  "wall_coordination_number":    42,
                                  "compaction_ratio": 0.5
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            650,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })

dempm.dem.add_attribute(materialID=2,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })
dempm.dem.add_attribute(materialID=3,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })

'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\taichi2025.2.25\GeoTaichi-main2025.2.25\example\dem\GranularPackings\sphere\SpherePacking.txt",
                               "GroupID": 0,
                               "MaterialID":2,
                               "InitialVelocity": ti.Vector([1.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free", "Free", "Free"],
                               "FixAngularVelocity": ["Free", "Free", "Free"]
                               }})
'''
'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Clump",
                               "GroupID": 0,
                               "MaterialID": 1,
                               "ClumpFile": r"E:\Taichi\clump\radius_small_50\clump.txt",
                               "PebbleFile": r"E:\Taichi\clump\radius_small_50\pebble.txt",
                               "InitialVelocity": ti.Vector([0,0.,0.]),
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
                                "ParticleFile": r"E:\Taichi\clump\gra_radius\radius0.02\radius4.4_20\particle.npz",
                                #"SphereFile": r"E:\Taichi\clump\gra_radius\radius0.4_4_50rock0.5_10clump\sphere.npz",
                                "ClumpFile": r"E:\Taichi\clump\gra_radius\radius0.02\radius4.4_20\clump.npz"
                               }})

dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=1,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.5,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=2,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.0,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=1,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=2,
                       materialID2=2,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=2,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.,
                           "Restitution": 0.0
                       })
dempm.dem.add_property(materialID1=3,
                       materialID2=1,
                       property={
                           "ShearModulus": 1e6,
                           "Possion": 0.3,
                           "Friction": 0.,
                           "Restitution": 0.0
                       })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\flume_small\left_front_behind.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\flume_small\flume-right.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\flume_small\flume-bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   3,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\flume_small\baffle-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\flume_small\flume-top.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   5,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\flume_small\flume-rightwater.stl",
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
'''
dempm.mpm.add_body_from_file(body={
                                    "FileType": "TXT",
                                    "Template":  {
                                                   "BodyID": 0,
                                                   "MaterialID": 1,
                                                   "ParticleFile": r"H:\particles\4450_001_114.txt"
                                                  }
                   })
'''
dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0, 0, 0]),
                            "BoundingBoxSize": ti.Vector([4, 0.6, 0.08]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])
dempm.mpm.add_region(region=[{
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0, 0, 0.08]),
                            "BoundingBoxSize": ti.Vector([2, 0.6, 0.2]),
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
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':0
                            }, dType='particle-wall')
dempm.add_property(DEMmaterial=3,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':0
                            }, dType='particle-wall')

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':0
                            }, dType='particle-particle')

dempm.run()
'''
dempm.dem.update_contact_properties(materialID1=3, materialID2=1, property_name="ShearModulus",value=0)
dempm.modify_parameters(SimulationTime=8)
dempm.run()
'''
dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

