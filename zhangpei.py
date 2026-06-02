
from geotaichi import *

init(device_memory_GB=35, debug=True)

dempm = DEMPM()
savepath=r"E:\taichi_results\ZP_TEST\ROCK30_0.03taylor_BBar"
dempm.set_configuration(domain=ti.Vector([101, 40, 30]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.1,
                      alphaPIC=0.085,
                      mapping="MUSL",
                      shape_function="GIMP",
                      #gravity=ti.Vector([4.905, 0., -8.496]),
                     #gravity=ti.Vector([7.515, 0., -6.306]),
                    #gravity=ti.Vector([6.937, 0., -6.937]),
gravity=ti.Vector([7.515, 0., -6.306]),
                      velocity_projection="Taylor", # 'Taylor'->TPIC, 'Affine'->APIC, PIC/FLIP'
                     stabilize="B-Bar Method",
                      material_type="Solid")# "B-Bar Method", "F-Bar Method"

dempm.dem.set_configuration(
                      boundary=["Reflect", "Reflect", "Reflect"],
                      gravity=ti.Vector([7.515, 0., -6.306]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      

dempm.set_solver({
                      "Timestep":         1e-4,
                      "SimulationTime":  3.0,
                      "SaveInterval":     0.5,
                      "SavePath":         savepath
                 })
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_particle_number": 1970,
                                "max_sphere_number": 1970,
                                "max_patch_number": 2330,
                                #"max_plane_number": 5,
                                #"max_facet_number": 5,
                                "compaction_ratio": 1.0,
                                "verlet_distance_multiplier":  0.15,
                                "body_coordination_number": 28,
                                "wall_coordination_number": 2000,
                                "wall_per_cell":6
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           374400,
                                "verlet_distance_multiplier":    0,
                                "max_constraint_number":  {

                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            

dempm.memory_allocate(memory={
                                  "body_coordination_number":    20,
                                  "wall_coordination_number":    2000,
                                  "compaction_ratio": 0.8
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.2,
                                "TorqueLocalDamping": 0.2
                            })
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.2,
                                "TorqueLocalDamping": 0.2
                            })

'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\zhangpei\zhangpei\lxp.txt",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Fix","Fix","Fix"],
                               "FixAngularVelocity": ["Fix","Fix","Fix"]
                               }})
'''
dempm.dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": r"E:\Taichi\zhangpei\zhangpei\after_rock30.txt",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})

dempm.dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")

dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                     "ShearModulus": 1e10,
                     "Possion": 0.3,
                     "Friction": 0.5,
                     "Restitution": 0.6
                 })

dempm.dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                     "ShearModulus": 1e10,
                     "Possion": 0.3,
                     "Friction": 0.5,
                     "Restitution": 0.6
                 })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   0,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-square_total.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   1,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-behind.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   2,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-bottom.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   3,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-front.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })

dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   4,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-left.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   5,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-right.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   6,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-top10.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
dempm.dem.add_wall(body={
        "WallType": "Patch",
        "WallID":   7,
        "MaterialID":   1,
        "Counterclockwise":  True,
        "WallFile":r"E:\Taichi\zhangpei\zhangpei\zp-temporarywall.stl",
        "Orientation":[0, 0, 1],
        "Translation": [0., 0., 0.],
        "Visualize": True
    })
'''


dempm.dem.select_save_data(particle=True, wall=True)

dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1.39e3,
                               "YoungModulus":                  2.e7,
                               "PoissionRatio":                 0.3,
                               "Friction":                      35,
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
                             "ElementSize":               ti.Vector([0.5, 0.5,0.5]),
                        })

dempm.mpm.add_region(region=[{
    "Name": "region1",
    "Type": "Rectangle",
    "BoundingBoxPoint": ti.Vector([2, 2,1]),
    "BoundingBoxSize": ti.Vector([22.5, 26, 10]),
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

dempm.mpm.select_save_data(particle=True)

dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")

dempm.add_property(DEMmaterial=0,
                 MPMmaterial=1,
                 property={
                     "ShearModulus": 1e9,
                     "Possion": 0.3,
                     "Friction": 1,
                     "Restitution": 0.69
                 })
dempm.add_property(DEMmaterial=1,
                 MPMmaterial=1,
                 property={
                     "ShearModulus": 1e9,
                     "Possion": 0.3,
                     "Friction": 0.5,
                     "Restitution": 0.69
                 })

dempm.run()

#dempm.dem.update_wall_status(wallID=6, property_name="Status", value=0)
#dempm.dem.update_wall_status(wallID=7, property_name="Status", value=0)
dempm.mpm.modify_parameters(SimulationTime=8.0,background_damping=0.05)
dempm.modify_parameters(SimulationTime=8.0,SaveInterval=0.03)
dempm.run()
'''
dempm.modify_parameters(SimulationTime=11.0,SaveInterval=0.03)
dempm.run()
'''
dempm.mpm.postprocessing(read_path=savepath)

dempm.dem.postprocessing(read_path=savepath)
