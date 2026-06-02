
from geotaichi import *

init(device_memory_GB=15, debug=False)

dempm = DEMPM()
savepath=r"E:\taichi_results\srrr22"
dempm.set_configuration(domain=ti.Vector([9, 5, 4]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.05,
                      alphaPIC=0.025,
                      mapping="MUSL",
                      shape_function="GIMP",
                      #gravity=ti.Vector([4.905, 0., -8.496]),
                     #gravity=ti.Vector([7.515, 0., -6.306]),
                    #gravity=ti.Vector([6.937, 0., -6.937]),
#gravity=ti.Vector([0, 0., -9.8]),
gravity=ti.Vector([8.496, 0., -4.905]),
                      velocity_projection="Taylor", # 'Taylor'->TPIC, 'Affine'->APIC, PIC/FLIP'
                     stabilize="B-Bar Method",
                      material_type="Solid")# "B-Bar Method", "F-Bar Method"

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([8.496, 0., -4.905]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         5e-4,
                      "SimulationTime":  0.1,
                    "SaveInterval" : 0.1,
                      "SavePath":         savepath
                 })
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 110,
                                "max_sphere_number": 110,
                                "max_patch_number": 23236,
                                #"max_plane_number": 5,
                                #"max_facet_number": 5,
                                "compaction_ratio": 1.0,
                                "verlet_distance_multiplier":  0.15,
                                "body_coordination_number": 28,
                                #"wall_coordination_number": 100,
                                "wall_per_cell":6
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           360000,
                                "verlet_distance_multiplier":    0,
                                "max_constraint_number":  {

                                                               "max_velocity_constraint":   3500000
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    20,
                                  "wall_coordination_number":    100,
                                  "compaction_ratio": 0.8
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            2600,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\impact_force_test\single_rock\single_rock.txt",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})

dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")

dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1,
                            "TangentialStiffness":        1,
                            "Friction":                   0.18,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
dempm.dem.add_property(materialID1=1,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e8,
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.3,
                            "TangentialViscousDamping":   0.3
                           })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\impact_force_test\single_rock\sr-behind.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\impact_force_test\single_rock\sr-bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\impact_force_test\single_rock\sr-right.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\impact_force_test\single_rock\zp-front.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\impact_force_test\single_rock\zp-left.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   5,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\impact_force_test\single_rock\sr-right11.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.select_save_data(particle=True, wall=True)

dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1.379e3,
                               "YoungModulus":                  2.e7,
                               "PoissionRatio":                 0.3,
                               "Friction":                      33,
                               "Dilation":                      0.0,
                               "Cohesion":                      0,
                               "Tensile":                       0
                 })
'''
dempm.mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2000.,
                               "YoungModulus":         1e9,
                               "PossionRatio":        0.0
                 })
'''
dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1,0.1]),
                        })

dempm.mpm.add_region(region=[{
    "Name": "region1",
    "Type": "Rectangle",
    "BoundingBoxPoint": ti.Vector([4., 3.,3.]),
    "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.1]),
    "zdirection": ti.Vector([0., 0., 1.])
}])

dempm.mpm.add_body(body={
    "Template": [{
        "RegionName": "region1",
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

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")
                            
dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            6,
                                 "TangentialStiffness":        4,
                                 "Friction":                   0.20,
                                 "NormalViscousDamping":       0.3,
                                 "TangentialViscousDamping":   0.3
                            })
dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            6,
                                 "TangentialStiffness":        4,
                                 "Friction":                   0.20,
                                 "NormalViscousDamping":       0.3,
                                 "TangentialViscousDamping":   0.3
                            })

dempm.run()
dempm.modify_parameters(SimulationTime=0.2,SaveInterval=0.01)
dempm.mpm.modify_parameters(SimulationTime=0.2,background_damping=0.02)
dempm.dem.update_wall_status(wallID=5, property_name="Status", value=0)
dempm.run()
dempm.mpm.postprocessing(read_path=savepath)

dempm.dem.postprocessing(read_path=savepath)
