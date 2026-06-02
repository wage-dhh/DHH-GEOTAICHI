from geotaichi import *

#init(device_memory_GB=45,debug=True, gpu_id=0)
init(arch="cpu", debug=True)

dempm = DEMPM()
outputdata=r'F:\taichi-results\radius\4times_sim\simulation4480_1'
dempm.set_configuration(domain=ti.Vector([10, 1, 5]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.01,
                      alphaPIC=0.01,
                      mapping="MUSL",
                      #shape_function="QuadBSpline",
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
                      "Timestep":         2e-4,
                      "SimulationTime":   0.5,
                      "CFL": 0.9,
                      "SaveInterval":     0.02,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 4,
                                "max_particle_number": 1750,
                                "max_sphere_number": 1,
                                "max_clump_number": 80,
                                "body_coordination_number": 40,
                                "wall_coordination_number": 42,
                                "max_facet_number": 42,
                                "wall_per_cell": 10,
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

dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 1,
                               "InitialVelocity": ti.Vector([0., 0., 0]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([1, 0.3, 1]),
                               "FixVelocity": ["Fix", "Fix", "Fix"],
                               "FixAngularVelocity": ["Fix", "Fix", "Fix"],
                               "Radius": 0.04,
                               "BodyOrientation": "uniform"}]})

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

dempm.dem.add_property(materialID1=1,
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
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([7, 0., 0.]),
                                    "vertice2": ti.Vector([7, 0., 1.2]),
                                    "vertice3": ti.Vector([7, 0.6, 1.2]),
                                    "vertice4": ti.Vector([7, 0.6, 0])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([0, 0.6, 0.]),
                                    "vertice3": ti.Vector([0, 0.6, 1.2]),
                                    "vertice4": ti.Vector([0, 0., 1.2])
                                   },
                   "OuterNormal": ti.Vector([1., 0., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([0, 0., 1.2]),
                                    "vertice3": ti.Vector([7, 0., 1.2]),
                                    "vertice4": ti.Vector([7, 0., 0])
                                   },
                   "OuterNormal": ti.Vector([0., 1., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0.6, 0.]),
                                    "vertice2": ti.Vector([7, 0.6, 0.]),
                                    "vertice3": ti.Vector([7, 0.6, 1.2]),
                                    "vertice4": ti.Vector([0, 0.6, 1.2])
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([7, 0., 0.]),
                                    "vertice3": ti.Vector([7, 0.6, 0.]),
                                    "vertice4": ti.Vector([0, 0.6, 0])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
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
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })

dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
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
                                       "InitialVelocity": ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]
                                       
                                   }]
                   })
'''
dempm.mpm.add_body_from_file(body={
                                    "FileType": "TXT",
                                    "Template":  {
                                                   "BodyID": 0,
                                                   "MaterialID": 1,
                                                   "ParticleFile": r"F:\taichi-results\4480_1_111.txt"
                                                  }
                   })
'''
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

dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

