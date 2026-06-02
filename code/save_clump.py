from geotaichi import *

init(gpu_id=1)

dem = DEM()


dem.set_configuration(domain=ti.Vector([6.,0.6,1]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([4,0.,0.]),#[2.,0.,0.]
                       "BoundingBoxSize": ti.Vector([1.1,0.6,0.5]),#[2.2,0.6,0.2]
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            


dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 21,
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
                                             "Position": ti.Vector([0., 0.02*6,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*7, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*7,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*8, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*8,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*9, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*9,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.02*10, 0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02*10,  0.]),
                                             "Radius": 0.02
                                            }]

                                 })
'''
dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 3,
                                 "Pebble": [{
                                             "Position": ti.Vector([0., 0,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.02,  0.]),
                                             "Radius": 0.02
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.02*1,  0.]),
                                             "Radius": 0.02
                                            }]

                                 })
'''
dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Clump",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               "Name": "clump1",
                               "MaxRadius": 0.04904800340580018,
                               "MinRadius": 0.04904800340580018,
                               "BodyNumber": 80,
                               #"ClumpPacking": r"E:\Taichi\clump\radius0.02_test\clump.txt",
                               "ClumpPacking": r"E:\Taichi\clump\radius0.02_4.4_80\clump.txt",
"PebblePacking": r"E:\Taichi\clump\radius0.02_4.4_80\pebble.txt",
                               "BodyOrientation": 'uniform'}})
                               #"BodyOrientation": 'constant'}})

