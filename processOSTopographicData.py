import os
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString


################################
#
# Initalise data directories and file names
#
###############################

with open("config.json") as f:
    config = json.load(f)

gis_data_dir = config['gis_data_dir']

qgis_workings_dir = os.path.join(gis_data_dir, "qgis_workings")

if os.path.isdir(qgis_workings_dir) == False:
    os.mkdir(qgis_workings_dir)
else:
    for f in os.listdir(qgis_workings_dir):
        os.remove(os.path.join(qgis_workings_dir, f))

output_directory = os.path.join(gis_data_dir, "processed_gis_data")

if os.path.isdir(output_directory) == False:
    os.mkdir(output_directory)


topographic_area_dir = os.path.join(gis_data_dir, config['topographic_area_dir'])
topographic_line_dir = os.path.join(gis_data_dir, config['topographic_line_dir'])

itn_link_file = os.path.join(output_directory, config["mastermap_link_processed_file"])
itn_node_file = os.path.join(output_directory, config["mastermap_node_processed_file"])

or_link_file = os.path.join(output_directory, config["openroads_link_processed_file"])
or_node_file = os.path.join(output_directory, config["openroads_node_processed_file"])


selection_layer_file = os.path.join(gis_data_dir, config['clip_file'])

topographic_area_file = os.path.join(topographic_area_dir, config['topographic_area_file'])
topographic_line_file = os.path.join(topographic_line_dir, config['topographic_line_file'])

output_vehicle_file = os.path.join(output_directory, config["topo_vehicle_processed_file"])
output_pedestrian_file = os.path.join(output_directory, config["topo_pedestrian_processed_file"])
output_line_file = os.path.join(output_directory, config["topo_boundary_processed_file"])

projectCRS = {'init' :'epsg:27700'}

priority_column = "priority"



#################################
#
#
# Functions
#
#
################################

# Disolved geometries are multi polygons, explode to single polygons
def explode(indf, single_type = Polygon, multi_type = MultiPolygon):
    data = {c:[] for c in indf.columns}
    outdf = gpd.GeoDataFrame(data)
    for idx, row in indf.iterrows():
        if type(row.geometry) == single_type:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == multi_type:
            multdf = gpd.GeoDataFrame(data)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf


#################################
#
# Read in the data
#
#################################

# Mastermap topographic area data  - for pavement and carriageway polygons
gdfTopoArea = gpd.read_file(topographic_area_file)
gdfTopoArea.crs = projectCRS

# Mastermap ITN data - for road network. Set columns to be those from the ITN raw data only, prevents duplicated columns when running this script multiple times
gdfITNLink = gpd.read_file(itn_link_file)
gdfITNLink.crs = projectCRS
gdfITNLink = gdfITNLink.reindex(columns = ['fid', 'version', 'versionDat', 'theme', 'changeDate', 'reasonForC','descriptiv', 'descript_1', 'natureOfRo', 'length', 'geometry'])

gdfITNNode = gpd.read_file(itn_node_file)
gdfITNNode.crs = projectCRS
gdfITNNode = gdfITNNode.reindex(columns = ['fid', 'version', 'versionDat', 'theme', 'changeDate', 'reasonForC','descriptiv', 'geometry'])

# OS Open Road - for ped road network
gdfORLink = gpd.read_file(or_link_file)
gdfORLink.crs = projectCRS
gdfORLink = gdfORLink.reindex(columns = ['fictitious', 'old_fid', 'class', 'roadNumber', 'name1','formOfWay', 'length', 'primary', 'trunkRoad','loop', 'structure', 'nameTOID', 'numberTOID', 'function', 'geometry','fid', 'MNodeFID', 'PNodeFID'])
gdfORNode = gpd.read_file(or_node_file)
gdfORNode.crs = projectCRS
gdfORNode = gdfORNode.reindex(columns = ['geometry', 'node_fid'])

# Study area polygon - to select data within the study area
gdfStudyArea = gpd.read_file(os.path.join(gis_data_dir, "study_area.shp"))
gdfStudyAreaWSG84 = gdfStudyArea.to_crs(epsg=4326)

studyPolygon = gdfStudyArea['geometry'].values[0]
studyPolygonWSG84 = gdfStudyAreaWSG84['geometry'].values[0]

################################
#
# Process topographic area data
#
################################

# Select vehicle areas as those that intersect the road network
# Results in 252 polygons, many more that previous method
or_plus_itn_link_geometries = pd.concat([gdfORLink.loc[:, ['geometry']], gdfITNLink.loc[:,['geometry']]])
gdfVehicle = gpd.sjoin(gdfTopoArea, or_plus_itn_link_geometries, op = 'intersects', rsuffix = 'links')
gdfVehicle.drop('index_links', axis = 1, inplace=True)
gdfVehicle.drop_duplicates(inplace = True)


#gdfVehicle = gdfTopoArea.loc[ (gdfTopoArea.geometry.intersects(gdfITNLink.geometry)) | (gdfTopoArea.geometry.intersects(gdfITNNode.geometry)) ]

# Select polygons with descriptive types that correspons to areas where pedestrians typically have priority
pedestrian_descriptivs = [  '(1:Path)','(2:Path,Structure)','(2:Path,Tidal Water)','(2:Roadside,Structure)','(1:Roadside)',
                            'Path','Path,Structure', 'Structure,Path', 'Path,Tidal Water','Roadside,Structure','Roadside', 'Roadside,Structure', 'Path,Roadside', 'Roadside,Path']
gdfPedestrian = gdfTopoArea.loc[gdfTopoArea['descriptiv'].isin(pedestrian_descriptivs)]

# Filter pedestrian polygons to just be those within the study area or that touch a vehicle polygon
gdfPedestrianA = gdfPedestrian.loc[ (gdfPedestrian.geometry.intersects(studyPolygon)) | (gdfPedestrian.geometry.within(studyPolygon))]

gdfPedestrianB = gdfPedestrian.copy()

# Loop through geometries and include them if they touch a vehicle polygon
keep_indices = []
for i, row in gdfPedestrianB.iterrows():
    poly = row['geometry']
    for veh_poly in gdfVehicle['geometry']:
        if poly.touches(veh_poly):
            keep_indices.append(i)
            break

gdfPedestrianB = gdfPedestrianB.loc[keep_indices]

# Overwrite initial ped polygon gdf with the polygons that are within the study area
gdfPedestrian = pd.concat([gdfPedestrianA, gdfPedestrianB]).drop_duplicates()

gdfPedestrianA = gdfPedestrianB = None

# Some pedestrian polygons get included in the vehicle polygon gdf because they intersect road links. Remove these
gdfVehicle = gdfVehicle.loc[ ~gdfVehicle['fid'].isin(gdfPedestrian['fid'])]

# Add in priority field
gdfPedestrian[priority_column] = "pedestrian"
gdfVehicle[priority_column] = "vehicle"


########################################
#
# Include any missed out pedestrian and vehicle polygons by finding the topographic polygons 
# that match up with interorior gaps in the pedestrian vehicle polygon set, that are of the descriptive 
# type that matches other ped and vehicle areas
#
########################################

# Combine pedestrian and vehicle areas into a single geo series
gdfPedVeh = pd.concat([gdfVehicle, gdfPedestrian])

gdfPedVeh['dissolve_key'] = 1
gdfDissolved = gdfPedVeh.dissolve(by = "dissolve_key")

gdfDissolved = explode(gdfDissolved)

interiors = []
for geom in gdfDissolved['geometry']:
    for i in geom.interiors:
        interiors.append(Polygon(i))

gdfPedVehInteriors = gpd.GeoDataFrame({'geometry':interiors})
gdfPedVehInteriors.crs = projectCRS

# Get Topographic polygons not included in the ped + veh selection
polyIDs = gdfPedVeh['fid'].values
gdfTopoAreaExcluded = gdfTopoArea.loc[ ~gdfTopoArea['fid'].isin(polyIDs)]

# Select those that intersect with the pedestrian and vehicle polygons, to see if they should have been included
# Loop through geometries and include them if they touch a vehicle polygon
candidate_indices = []
for i, row in gdfTopoAreaExcluded.iterrows():
    poly = row['geometry']
    for pv_poly in gdfPedVehInteriors['geometry']:
        if poly.intersects(pv_poly):
            candidate_indices.append(i)
            break
gdfTopoAreaCandidates = gdfTopoAreaExcluded.loc[candidate_indices]

gdfVehicleAdditions  = gdfTopoAreaCandidates.loc[ gdfTopoAreaCandidates['descriptiv'] == "(1:Road Or Track)"]
gdfPedestrianAdditions  = gdfTopoAreaCandidates.loc[ gdfTopoAreaCandidates['descriptiv'].isin(pedestrian_descriptivs)]

gdfVehicleAdditions[priority_column] = 'vehicle'
gdfPedestrianAdditions[priority_column] = 'pedestrian'

gdfPedVeh = pd.concat([gdfPedVeh, gdfPedestrianAdditions, gdfVehicleAdditions])


########################################
#
# Select only polygons that are within the largest connected component, avoids inaccessible islands
#
########################################

# initialise df to recod the neighbours for each ped-vehicle space polygon. use to identify clusters
dfPedVehNeighbours = pd.DataFrame()
for index, row in gdfPedVeh.iterrows():
    # Touches identifies polygons with at least one point in common but interiors don't intersect. So will work as long as none of my topographic polygons intersect
    neighborFIDs = gdfPedVeh[gdfPedVeh.geometry.touches(row['geometry'])]['fid'].tolist() 
    #neighborFIDs = neighborFIDs.remove(row['fid']) polygons don't touch them selves because they intersect
    df = pd.DataFrame({'neighbourfid':neighborFIDs})
    df['fid'] = row['fid']
    dfPedVehNeighbours = pd.concat([dfPedVehNeighbours, df])


g = nx.from_pandas_edgelist(dfPedVehNeighbours, source='fid', target = 'neighbourfid')
ccs = list(nx.connected_components(g))
ccs.sort(key=len)

cc_largest = ccs.pop(-1)


# Ideally want to restrict analysis to just the largest connected component since other ccs are likely to be court yard or other areas separate from roads
# For the smaller ccs check that they dont intersect with road network, in which case they can be excluded
# (might also want to check the road network is a single connected component)

check_ok = True
for cc in ccs:
    gdfCC = gdfPedVeh.loc[ gdfPedVeh['fid'].isin(cc)]

    # Check that none of these intersect with road links
    if gpd.sjoin(gdfCC, gdfORLink, op='intersects').shape[0] != 0:
        print("Connected component at index {} intersects the road network".format(ccs.index(cc)))
        check_ok = False

assert check_ok


# Filter pedestrian and vehicle polygons to just include those in the largest connected component
gdfPedVeh = gdfPedVeh.loc[ gdfPedVeh['fid'].isin(cc_largest)]

# Overwrite previous pedestrian and vehicle polygon geo dataframes as will want to replace witht he filtered data
gdfPedestrian = gdfVehicle = None


gdfPedestrian = gdfPedVeh.loc[ gdfPedVeh[priority_column] == 'pedestrian']
gdfVehicle = gdfPedVeh.loc[ gdfPedVeh[priority_column] == 'vehicle']


# Check that no polygons appear in both ped and veh dataframes
assert len(set(gdfPedestrian['fid']).intersection(set(gdfVehicle['fid']))) == 0


###########################
#
#
# Linking Pedestrian and Vehicle polygons with Road Link IDs
#
#
###########################

# Buffer road node
node_buffer_dist = 0.01
gdfORNodeBuffer = gdfORNode.copy()
gdfORNodeBuffer['geometry'] = gdfORNode.buffer(node_buffer_dist)

# Clip road links to exclude the junctions
gdfORLinkClipped = gpd.overlay(gdfORLink, gdfORNodeBuffer, how = 'difference')

# Save to file so can be read into QGIS layer
output_or_link_clipped_file = os.path.join(qgis_workings_dir, "or_link_clipped.shp")
gdfORLinkClipped.to_file(output_or_link_clipped_file)


# Initialise QGIS API
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



or_links_clipped = QgsVectorLayer(output_or_link_clipped_file, "or_links_clipped", "ogr")

'''
# Loop through features and densify
densify_distance = 5 # Points separated by 5m
for f in or_links_clipped.getFeatures():
    g = f.geometry()
    newG = g.densifyByDistance(1)
    f.setGeometry(newG)
'''

# Extract points
points_path = os.path.join(qgis_workings_dir, "or_links_dense_clipped_points.shp")
fields = or_links_clipped.fields()
fields.append(QgsField("point_id",  QVariant.Int))
writer = QgsVectorFileWriter(points_path, "UTF-8", fields, QgsWkbTypes.Point, driverName="ESRI Shapefile")

if writer.hasError() != QgsVectorFileWriter.NoError:
	print("Error when creating shapefile: ",  writer.errorMessage())

# Loop through the densified cliped linestrings and extract the points
point_id = 0
for f in or_links_clipped.getFeatures():
    g = f.geometry()
    gD = g.densifyByDistance(0.2)
    for p in gD.vertices():
        output_feature = QgsFeature()
        attrs = f.attributes()
        attrs.append(point_id)
        output_feature.setAttributes(attrs)
        output_feature.setGeometry(p)
        writer.addFeature(output_feature)
del writer

or_links_dense_clipped_points = QgsVectorLayer(points_path, "or_links_dense_clipped_points", "ogr")

# Now get voronoi polygons from these points
outpath = os.path.join(qgis_workings_dir, "or_links_voronoi.shp")
result = processing.run("qgis:voronoipolygons", {"INPUT":or_links_dense_clipped_points, "BUFFER":100, "OUTPUT":outpath})

# Read in voronoi polygons as geodataframe
gdfLinksVoronoi = gpd.read_file(outpath)
gdfLinksVoronoi.crs = projectCRS

# Disolve by road link fid to get road link voroni regions
gdfLinksVoronoi = gdfLinksVoronoi.dissolve(by = 'fid')
gdfLinksVoronoi.reset_index(inplace=True)
gdfLinksVoronoi.to_file(os.path.join(qgis_workings_dir, "or_links_voronoi_dissolve.shp"))


# Intersect pedestrian and vehicle priority polygons with links voronoi and disolve by road link fid
gdfPedestrianLinks = gpd.overlay(gdfPedestrian, gdfLinksVoronoi, how = 'intersection')

# Duplicate columns result in suffixes, edit these to make more sense
rename_dict = {i:i.replace('_1','_topo').replace('_2','_or') for i in gdfPedestrianLinks.columns}
gdfPedestrianLinks.rename(columns = rename_dict, inplace = True)

# Dissolve and explode so only single non intersecting polygons per road link
gdfPedestrianLinks = gdfPedestrianLinks.dissolve(by = 'fid_or').reset_index()
gdfPedestrianLinks = explode(gdfPedestrianLinks)

# Create new ped poly id and drop old fid field which is now not a useable id
gdfPedestrianLinks = gdfPedestrianLinks.drop(['fid_topo'], axis = 1)
gdfPedestrianLinks['polyID'] = ["ped_poly_id_{}".format(i) for i in gdfPedestrianLinks.index]
assert gdfPedestrianLinks['polyID'].duplicated().any()  == False
gdfPedestrianLinks.crs = gdfPedestrian.crs


# Do the same for vehicle polys
gdfVehicleLinks = gpd.overlay(gdfVehicle, gdfLinksVoronoi, how = 'intersection')
rename_dict = {i:i.replace('_1','_topo').replace('_2','_or') for i in gdfVehicleLinks.columns}
gdfVehicleLinks.rename(columns = rename_dict, inplace = True)
gdfVehicleLinks = gdfVehicleLinks.dissolve(by = 'fid_or').reset_index()
gdfVehicleLinks = explode(gdfVehicleLinks)

# Create new ped poly id and drop old fid field which is now not a useable id
gdfVehicleLinks = gdfVehicleLinks.drop(['fid_topo'], axis = 1)
gdfVehicleLinks['polyID'] = ["veh_poly_id_{}".format(i) for i in gdfVehicleLinks.index]
assert gdfVehicleLinks['polyID'].duplicated().any()  == False
gdfVehicleLinks.crs = gdfVehicle.crs

# Rename the ITN Road Link FID column to match the name expected by the Repast Model
gdfVehicleLinks.rename(columns = {"fid_or":"roadLinkID"}, inplace = True)
gdfPedestrianLinks.rename(columns = {"fid_or":"roadLinkID"}, inplace = True)


##################################
#
# Process topographic line data
#
#
# Have decided to use the perimiter of the pedestrian and vehicle polygons combined. This is less restrictive that the topographic lines
# that are categorised as obstructing, which in some places wouls unrealistically constrict pedestrian movement.
#
##################################

# Get perimiter(s) of pedestrian and vehicle areas and include this with the obstructing lines (will need to make sure that this perimiter includes all ITN network)
gdfPedVehLinks = pd.concat([gdfVehicleLinks, gdfPedestrianLinks])

# Disolve pedestrian and vehicle polygons into single polygon and extract perimiter
gdfPedVehLinks['dissolve_key'] = 1
gdfDissolved = gdfPedVehLinks.dissolve(by = "dissolve_key")
gdfDissolved = explode(gdfDissolved)

# Get linstrings of exterior and interior of the dissolved pedestrian + vehicle polygons. These will mark the perimiters of the space
data = {'type':[], 'geometry':[]}
gdfPerimiter = gpd.GeoDataFrame(data)
gdfPerimiter.crs = gdfDissolved.crs
for geom in gdfDissolved['geometry']:
    exterior = LineString(geom.exterior.coords)
    gdfPerimiter = gdfPerimiter.append({"type":"exterior", "geometry":exterior}, ignore_index = True)

    for i in geom.interiors:
        interior = LineString(i)
        gdfPerimiter = gdfPerimiter.append({"type":"interior", "geometry":interior}, ignore_index = True)

gdfPerimiter.crs = projectCRS
gdfPerimiter[priority_column] = "pedestrian_obstruction"

# Check how many exterior boundaries - ideally want one by can have more than this is largest connected component of ped&veh polys includes a polygon
# that only touches at one coordinate
print(gdfPerimiter['type'].value_counts())


##################################
#
#
# Lookup from ITN road link ID or OR road link ID
#
# Ped and vehicle polygons are linked to the open roads links since these are used for pedestrian routing.
# However, a pedestrian needs to identify number of vehciles on a road. Vehicle counts are aggregated to ITN road link.
# To find number of vehicles on a road link segment need to add open road ID to ITL road links so they can be identified as belonging to a road section
# Do this by:
# - find proportion of itn link that belongs in each vehicle polygon it intersects
# - assign it the ped road link id of the vehicle poly it most intersects with
#
#################################

# Calculate the linestring lengths
gdfITNLinkInt = gdfITNLink.copy()
gdfITNLinkInt['tot_length'] = gdfITNLinkInt['geometry'].map(lambda g: g.length)

# Intersect ITN links with vehicle polygons
gdfITNLinkInt = gpd.overlay(gdfITNLinkInt, gdfVehicleLinks, how = 'intersection')

# Calculate lengths again
gdfITNLinkInt['length_intersect'] = gdfITNLinkInt['geometry'].length
gdfITNLinkInt['proportion'] = gdfITNLinkInt['length_intersect'] / gdfITNLinkInt['tot_length']

# Now group by itn link id and calculate percentage length in each bit and assign ped road link id based on max proportion
gdfITNLinkPedRLID = gdfITNLinkInt.groupby('fid').apply(lambda df: df.sort_values(by = 'proportion', ascending = False).head(1))
gdfITNLinkPedRLID.index = np.arange(gdfITNLinkPedRLID.shape[0])

# Group by itn link ID and ped road link id to calculate length in 
gdfITNLinkPedRLID = gdfITNLinkPedRLID.reindex(columns = ['fid', 'roadLinkID', 'proportion'])
gdfITNLinkPedRLID.rename(columns = {'roadLinkID':'pedRLID'}, inplace = True)

# Check all links still present and proportions reasonably high
assert gdfITNLinkPedRLID['fid'].duplicated().any() == False
assert len(gdfITNLinkPedRLID['fid'].unique()) == len(gdfITNLink['fid'].unique()) # Fails since ITN road links now extend beyond ped study area
assert gdfITNLinkPedRLID['proportion'].min() > 0.75 #FAILS

# Join lookup from fid to ped road link id to original ITN Link Data
gdfITNLink = gdfITNLink.merge(gdfITNLinkPedRLID, on = 'fid')

# For the OR links the ped Road Link is is given by the fid since these links used to distinguish between road polygons.
gdfORLink['pedRLID'] = gdfORLink['fid']


###########################
#
# Save processed data
#
###########################
# Save these polygon layers
gdfPedestrianLinks.to_file(output_pedestrian_file)
gdfVehicleLinks.to_file( output_vehicle_file)
gdfPerimiter.to_file(output_line_file)

gdfITNLink.to_file(itn_link_file)
gdfORLink.to_file(or_link_file)

'''
# Now densify the clipped linestrings
def densify(l, d):
    length = l.length

    if l.length < d:
        return l

    # Find minimum number of points needed between start and end to achieve max separation of d
    np = int(length / d)
    dp = length / (np + 1) # total length divided by number of gaps

    points = []
    points.append(l.coords[0])
    dtot = dp

    # number of points to add in is np
    for i in range(np):
        points.append(l.interpolate(dp*i))

    # Add final point
    points.append(l.coords[-1])
    
    # Return densified linestring
    return LineString(points)


'''