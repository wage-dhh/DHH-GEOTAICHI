from geotaichi import *

init()

dem = DEM()


dem.set_configuration(domain=ti.Vector([10.,3.,4]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([1.9,0.,0.]),
                       "BoundingBoxSize": ti.Vector([2.2,0.6,0.2]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            
                          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               
                               "MaxRadius": 0.005,
                               "MinRadius": 0.005,
                               "BodyNumber": 135168}})

