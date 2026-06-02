from geotaichi import *

init(device_memory_GB=25,debug=False,gpu_id=1)

dempm = DEMPM()
outputdata=r'H:\taichi_results\Tsunami\Tsunami0.013'
dempm.set_configuration(domain=ti.Vector([5, 0.6, 3]),
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
                      gravity=ti.Vector([0, 0., -9.8]))

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([0, 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         2e-4,
                      "SimulationTime":   5,
                      "CFL": 0.85,
                      "SaveInterval":     0.05,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 5,
                                "max_particle_number": 154,
                                "max_sphere_number": 154,
                                "max_clump_number": 0,
                                "max_patch_number": 14,
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
                                  "body_coordination_number":    50,
                                  "wall_coordination_number":    14,
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
                                "Density":            2500,
                                "ForceLocalDamping":  0.3,
                                "TorqueLocalDamping": 0.3
                            })


dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\rock\tsunami.txt",
                               "GroupID": 0,
                               "MaterialID":1,
                               "InitialVelocity": ti.Vector([0.866,0.,-0.5]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free", "Free", "Free"],
                               "FixAngularVelocity": ["Free", "Free", "Free"]
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
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   0,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\tsunami.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''
dempm.dem.add_wall(body={
                   "WallType":    "Facet",
                   "MaterialID":   0,
                   "WallShape":   "Polygon",
                   "WallID":      0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([1.8, 0., 0.]),
                                    "vertice2": ti.Vector([4., 0, 0.]),
                                    "vertice3": ti.Vector([4, 0.6, 0.]),
                                    "vertice4": ti.Vector([1.8, 0.6, 0.])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
                  })
dempm.dem.add_wall(body={
                   "WallType":    "Facet",
                   "MaterialID":   0,
                   "WallShape":   "Polygon",
                   "WallID":      1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([0., 0, 2.]),
                                    "vertice3": ti.Vector([4, 0., 2.]),
                                    "vertice4": ti.Vector([4, 0., 0.])
                                   },
                   "OuterNormal": ti.Vector([0, 1., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallType":    "Facet",
                   "MaterialID":   0,
                   "WallShape":   "Polygon",
                   "WallID":      2,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0.6, 0.]),
                                    "vertice2": ti.Vector([4., 0.6, 0.]),
                                    "vertice3": ti.Vector([4, 0.6, 2]),
                                    "vertice4": ti.Vector([0, 0.6, 2.])
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallType":    "Facet",
                   "MaterialID":   0,
                   "WallShape":   "Polygon",
                   "WallID":      3,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([4, 0., 0.]),
                                    "vertice2": ti.Vector([4., 0, 2]),
                                    "vertice3": ti.Vector([4, 0.6, 2]),
                                    "vertice4": ti.Vector([4, 0.6, 0.])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallType":    "Facet",
                   "MaterialID":   0,
                   "WallShape":   "Polygon",
                   "WallID":      1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([1.99, 0., 0.]),
                                    "vertice2": ti.Vector([1.99, 0.6, 0.]),
                                    "vertice3": ti.Vector([0, 0.6, 1.1447]),
                                    "vertice4": ti.Vector([0, 0., 1.1447])
                                   },
                   "OuterNormal": ti.Vector([0.5, 0., 0.866])
                  })
'''

dempm.dem.select_save_data(clump=True,particle=True, wall=True)

dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1200.,
                               "Modulus":              1e6,
                               "Viscosity":            0.001,
                 })
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.013, 0.013, 0.013])
                        })
dempm.mpm.add_body_from_file(body={
                       "FileType": "TXT",
                       "Template": [{
                                       "ParticleFile":r"E:\Taichi\rock\mpm_particle\0.013after.txt",
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
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':0
                            }, dType='particle-wall')

dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e5,
                                 "NormalViscousDamping":          0.3,
'NonSlipFraction':1
                            }, dType='particle-particle')


dempm.run()
'''
dempm.dem.update_wall_status(wallID=4, property_name="Status", value=0)
dempm.dem.update_wall_status(wallID=5, property_name="Status", value=0)
dempm.dem.modify_parameters(SimulationTime=20,gravity=ti.Vector([1.7, 0., -9.65]))

dempm.modify_parameters(SimulationTime=20.0,SaveInterval=0.001)
dempm.run()
'''
dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

