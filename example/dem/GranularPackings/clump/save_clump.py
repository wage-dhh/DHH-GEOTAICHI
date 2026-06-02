from geotaichi import *

init()

dem = DEM()


dem.set_configuration(domain=ti.Vector([2.,2,3]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0,0,2]),
                       "BoundingBoxSize": ti.Vector([1,1,1]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            


dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 8,
                                 "Pebble": [{
                                             "Position": ti.Vector([0., 0.,  0.]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0.1, 0,  0.]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0.,  0.1, 0.]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0.1, 0.1,  0.]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.,  0.1]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0.1, 0.0, 0.1]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.1, 0.1]),
                                             "Radius": 0.1
                                            },
                                     {
                                             "Position": ti.Vector([0.1,  0.1, 0.1]),
                                             "Radius": 0.1
                                            }]

                                 })

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Clump",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               "Name": "clump1",
                               "MaxRadius": 0.16638148952632292,
                               "MinRadius": 0.16638148952632292,
                               "BodyNumber": 200,
                               "ClumpPacking": r"F:\taichi\clump\radius1.0_1_1cube\clump.txt",
                               "PebblePacking": r"F:\taichi\clump\radius1.0_1_1cube\pebble.txt",
                               "BodyOrientation": 'uniform'}}) 

