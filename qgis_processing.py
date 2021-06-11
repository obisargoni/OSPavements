import os
import sys
from qgis.core import *

from qgis.analysis import QgsNativeAlgorithms
from qgis.PyQt.QtCore import QVariant

# Initialise QGIS
qgis_dir = "C:\\Program Files\\QGIS 3.4\\"
QgsApplication.setPrefixPath(os.path.join(qgis_dir, "bin\\qgis-ltr-bin-g7.exe"), True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Append the path where processing plugin can be found
sys.path.append(os.path.join(qgis_dir, '\\apps\\qgis-ltr\\python\\plugins'))


from qgis import processing
from processing.core.Processing import Processing
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

#######################################
#
#
# Paths and globals
#
#
#######################################

gis_data_dir = "S:\\CASA_obits_ucfnoth\\1. PhD Work\\GIS Data\\CoventGardenWaterloo"

output_directory = os.path.join(gis_data_dir, "processed_gis_data")

if os.path.isdir(output_directory) == False:
    os.mkdir(output_directory)


output_vehicle_file = os.path.join(output_directory, "topographicAreaVehicle.shp")
output_pedestrian_file = os.path.join(output_directory, "topographicAreaPedestrian.shp")

output_itn_link_file = os.path.join(output_directory, "mastermap-itn RoadLink Intersect Within.shp")
output_itn_node_file = os.path.join(output_directory, "mastermap-itn RoadNode Intersect Within.shp")


projectCRS = {'init' :'epsg:27700'}

priority_column = "priority"


#######################################
#
#
# Load pedestrian and vehicle layers
#
#
#######################################

pedestrian_polygons = QgsVectorLayer(output_pedestrian_file, "pedestrian_polygons", "ogr")
vehicle_polygons = QgsVectorLayer(output_vehicle_file, "vehicle_polygons", "ogr")

itn_links = QgsVectorLayer(output_itn_link_file, "itn_links", "ogr")
itn_nodes = QgsVectorLayer(output_itn_node_file, "itn_nodes", "ogr")


######################################
#
#
# Identify and run algorithms
#
# Want to get voronoi regions for road link lines
#
# Use these to link pedestrian and vehicle polygons to road links
#
#######################################

def alg_search(term = ''):
    for alg in QgsApplication.processingRegistry().algorithms():
        if term.lower() in str(alg.displayName).lower():
            print("{}:{} --> {}".format(alg.provider().name(), alg.name(), alg.displayName()))

qgis_workings_dir = os.path.join(gis_data_dir, "qgis_workings")

if os.path.isdir(qgis_workings_dir) == False:
    os.mkdir(qgis_workings_dir)

# Buffer ITN Nodes
outpath = os.path.join(qgis_workings_dir, "nodes_buffer.shp")
result = processing.runAndLoadResults("gdal:buffervectors", {'INPUT': itn_nodes, 'DISTANCE':1, 'OUTPUT':outpath})

itn_nodes_buffer = QgsProject.instance().mapLayersByName("nodes_buffer")[0]


# Densify ITN Links

# copy itn_links layer
itn_links_dense = itn_links.clone()

# Loop through features and densify
densify_distance = 5 # Points separated by 5m
for f in itn_links_dense.getFeatures():
    g = f.geometry()
    newG = g.densifyByDistance(5)
    f.setGeometry(newG)


# Clip densified links by buffered nodes
outpath = os.path.join(qgis_workings_dir, "itn_links_dense_clipped.shp")
result = processing.runAndLoadResults("gdal:clipvectorbypolygon", {"INPUT":itn_links_dense, "MASK":itn_nodes_buffer,
                                                                    "OUTPUT":outpath})
itn_links_dense_clipped = QgsProject.instance().mapLayersByName("itn_links_dense_clipped")[0]

# Get points from dense clipped links
# create layer
itn_links_dense_clipped_points = QgsVectorLayer("Point", "itn_links_dense_clipped_points", "memory")
pr = itn_links_dense_clipped_points.dataProvider()
# add fields
pr.addAttributes(itn_links_dense_clipped.fields())
pr.addAttributes([QgsField("point_id",  QVariant.Int)])
itn_links_dense_clipped_points.updateFields() # tell the vector layer to fetch changes from the provider

# Loop through the densified cliped linestrings and extract the points
point_id = 0
for f in itn_links_dense_clipped.getFeatures():
    g = f.geometry().get()
    print(g)
    attrs = f.attributes()
    for p in g.vertices():
        output_feature = QgsFeature()
        pAttrs = attrs.append(point_id)
        output_feature.setAttributes(pAttrs)
        output_feature.setGeometry(p)
        print(output_feature)
        res = pr.addFeature(output_feature)
        print(res)

# update layer's extent when new features have been added
# because change of extent in provider is not propagated to the layer
itn_links_dense_clipped_points.updateExtents()



