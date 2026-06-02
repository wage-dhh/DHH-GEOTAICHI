from geotaichi import *

init(device_memory_GB=45,debug=False,gpu_id=1)

dempm = DEMPM()
outputdata=r'H:\taichi_results\dambreak\dambreak'
dempm.set_configuration(domain=ti.Vector([0.584, 0.2, 1]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.005,
                      alphaPIC=0.,
                      mapping="MUSL",
                      shape_function="QuadBSpline",
material_type="Fluid",
                      gravity=ti.Vector([0, 0., -9.8]))

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([0, 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         2e-5,
                      "SimulationTime":   1,
"CFL": 0.9,
                      "SaveInterval":     0.2,
                      "SavePath":         outputdata
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_particle_number": 2,
                                "max_sphere_number": 2,
                                "max_clump_number": 0,
"max_facet_number": 10,
"wall_per_cell":6,
                                "compaction_ratio": 0.4,
                                "verlet_distance_multiplier":  0.15,
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           8526400,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0,
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    17,
                                  "wall_coordination_number":    10,
                                  "compaction_ratio": 0.4
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
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            860,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0., 0., 0]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([0.05, 0.1, 0.33]),
                               "FixVelocity": ["Fix", "Fix", "Fix"],
                               "FixAngularVelocity": ["Fix", "Fix", "Fix"],
                               "Radius": 0.01,
                               "BodyOrientation": "uniform"}]})
dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                               particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=0,
                       materialID2=0,
                       property={
                           "ShearModulus": 1e5,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.01
                       })

dempm.dem.add_property(materialID1=0,
                       materialID2=1,
                       property={
                           "ShearModulus": 5e5,
                           "Possion": 0.3,
                           "Friction": 0.7,
                           "Restitution": 0.0
                       })


'''
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\wall\dambreak\flume_total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''
dempm.dem.add_wall(body={
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([0, 0., 0.7]),
                                    "vertice3": ti.Vector([0.584, 0., 0.7]),
                                    "vertice4": ti.Vector([0.584, 0., 0.])
                                   },
                   "OuterNormal": ti.Vector([0., 1., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0.2, 0.]),
                                    "vertice2": ti.Vector([0.584, 0.2, 0.]),
                                    "vertice3": ti.Vector([0.584, 0.2, 0.7]),
                                    "vertice4": ti.Vector([0., 0.2, 0.7])
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([0., 0.2, 0.]),
                                    "vertice3": ti.Vector([0., 0.2, 0.7]),
                                    "vertice4": ti.Vector([0., 0., 0.7])
                                   },
                   "OuterNormal": ti.Vector([1, 0., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.584, 0., 0.]),
                                    "vertice2": ti.Vector([0.584, 0., 0.7]),
                                    "vertice3": ti.Vector([0.584, 0.2, 0.7]),
                                    "vertice4": ti.Vector([0.584, 0.2, 0.])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  })
dempm.dem.add_wall(body={
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0, 0., 0.]),
                                    "vertice2": ti.Vector([0.584, 0., 0.]),
                                    "vertice3": ti.Vector([0.584, 0.2, 0.]),
                                    "vertice4": ti.Vector([0., 0.2, 0.])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
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
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.0]),
                            "BoundingBoxSize": ti.Vector([0.146, 0.2, 0.292]),
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
dempm.add_body(check_overlap=False)

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
'NonSlipFraction':0.01
                            }, dType='particle-wall')

dempm.run()

dempm.mpm.postprocessing(read_path=outputdata)

dempm.dem.postprocessing(read_path=outputdata)

