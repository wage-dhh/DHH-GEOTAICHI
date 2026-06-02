from geotaichi import *

init(device_memory_GB=45,debug=False, gpu_id=1)

dempm = DEMPM()
outputdata=r'H:\taichi_results\radius\radius0.04\radius_small20'
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
                                "max_sphere_number": 350,
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
                                  "body_coordination_number":    100,
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
                                "Density":            2650,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\rock\two_rock.txt",
                               "GroupID": 0,
                               "MaterialID":1,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Fix", "Fix", "Fix"],
                               "FixAngularVelocity": ["Fix", "Fix", "Fix"]
                               }})
'''

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Clump",
                               "GroupID": 0,
                               "MaterialID": 1,
                               "ClumpFile": r"E:\Taichi\clump\radius_small\clump.txt",
                               "PebbleFile": r"E:\Taichi\clump\radius_small\pebble.txt",
                               "InitialVelocity": ti.Vector([0,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})
'''
'''
dempm.dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 19,
                                 "Pebble": [{
                                             "Position": ti.Vector([0., 0,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  0.02*2,0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*3,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.02*1,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*2, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*3, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*4,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*5,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.02*4,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.02*5,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*6, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*7, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*6,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*7,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*8,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.02*8,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*9,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.02*9,  0.]),
                                             "Radius": 0.02
                                            }]

                                 })

dempm.dem.create_body(body={
    "GenerateType": "Create",
    "BodyType": "Clump",
    "Template": [{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.2, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },
{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.3, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.4, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.5, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },
{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.6, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },
{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.7, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.8, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.9, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [5.0, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [5.1, 0.3, 0.05],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.2, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },
{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.3, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.4, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.5, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },
{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.6, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },
{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.7, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.8, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [4.9, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [5.0, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    },{
        "Name": "clump1",
        "GroupID": 0,
        "MaterialID": 1,
        "BodyPoint": [5.1, 0.3, 0.15],
        "Radius": 1.0 * 0.04747393012644594,
        "BodyOrientation": "constant"
    }
    ]})
'''
dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=1,
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
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-right.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\baffle-total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\end_experiments_small\flume-top.stl",
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
                               "Viscosity":            0.001,
                 })
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.025, 0.025, 0.025])
                        })

dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 0, 0]),
                            "BoundingBoxSize": ti.Vector([2.2, 0.6, 0.2]),
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
                                       "InitialVelocity":ti.Vector([1, 0, 0]),
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
                            }, dType='particle-particle')

dempm.run()

dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

