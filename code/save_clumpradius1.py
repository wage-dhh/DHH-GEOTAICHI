from geotaichi import *

init()

dem = DEM()


dem.set_configuration(domain=ti.Vector([60.,6,10]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([32,0.,0]),
                       "BoundingBoxSize": ti.Vector([20,6,5]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            


dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 7,
                                 "Pebble": [{
                                             "Position": ti.Vector([0., 0,  0.]),
                                             "Radius": 0.5
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.5,  0.]),
                                             "Radius": 0.5
                                            },
                                     {
                                             "Position": ti.Vector([0.,  0.5*2,0.]),
                                             "Radius": 0.5
                                            },
                                     {
                                             "Position": ti.Vector([0., 0.5*3,  0.]),
                                             "Radius": 0.5
                                            },
                                     {
                                             "Position": ti.Vector([0., -0.5*1,  0.]),
                                             "Radius": 0.5
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.5*2, 0.]),
                                             "Radius": 0.5
                                            },
                                     {
                                             "Position": ti.Vector([0.,  -0.5*3, 0.]),
                                             "Radius": 0.5
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
                               "MaxRadius": 0.8620398395878553,
                               "MinRadius": 0.8620398395878553,
                               "BodyNumber": 30,
                               "ClumpPacking": r"E:\Taichi\clump\radius1\clump.txt",
                               "PebblePacking": r"E:\Taichi\clump\radius1\pebble.txt",
                               "BodyOrientation": 'uniform'}}) 

