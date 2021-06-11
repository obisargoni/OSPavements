import osmnx
import geopandas as gpd
import json

projectCRS = "epsg:27700"

with open("config.json") as f:
    config = json.load(f)

gis_data_dir = config['gis_data_dir']

poi_file = os.path.join(gis_data_dir, config['poi_file'])

gdfPOIs = gpd.read_file(poi_file)
if gdfITNNode.crs is None:
    gdfITNNode.crs = projectCRS
else:
    assert gdfITNNode.crs.to_string().lower() == projectCRS

# Study area polygon - to select data within the study area

centre_poi = gdfPOIs.loc[gdfPOIs['ref_no'] == config['centre_poi_ref']] 
centre_poi_geom = centre_poi['geometry'].values[0]

seriesStudyArea = centre_poi.buffer(config['study_area_dist'])
seriesStudyArea.to_file(os.path.join(gis_data_dir, "study_area.shp"))
gsStudyAreaWSG84 = seriesStudyArea.to_crs(epsg=4326)

studyPolygon = seriesStudyArea.values[0]
studyPolygonWSG84 = gsStudyAreaWSG84.values[0]

################################
#
# alternatively use osmnx to select open road network
#
################################

# This point based mathod doesn't worth for some reason
'''
gdfPOIsWSG84 = gdfPOIs.to_crs(epsg=4326)
centre_point = gdfPOIsWSG84.loc[gdfPOIsWSG84['ref_no'] == config['centre_poi_ref'], 'geometry'].values[0].coords[0]
open_network = osmnx.graph.graph_from_point(centre_point, dist=config['study_area_dist'], dist_type='bbox', network_type='all', simplify=True, retain_all=False, truncate_by_edge=True, clean_periphery=True, custom_filter=None)
'''

# Try using polygon instead
open_network = osmnx.graph.graph_from_polygon(studyPolygonWSG84, network_type='all', simplify=True, retain_all=False, truncate_by_edge=True, clean_periphery=True, custom_filter=None)

# Get undirected non multi graph version
#D = osmnx.get_digraph(open_network) # Converts from multi di graph to di graph
U = open_network.to_undirected()

gdf_nodes, gdf_edges = osmnx.graph_to_gdfs(U)
gdf_edges.reset_index(inplace = True)
gdf_nodes = gdf_nodes.to_crs(projectCRS)
gdf_edges = gdf_edges.to_crs(projectCRS)

# Find node closest to centre OR node
gdf_nodes['dist_to_centre'] = gdf_nodes.distance(centre_poi_geom)
nearest_node_id = gdf_nodes.sort_values(by = 'dist_to_centre', ascending=True).index[0]

reachable_nodes = largest_connected_component_nodes_within_dist(U, nearest_node_id, config['study_area_dist'], 'length')

gdf_nodes = gdf_nodes.loc[reachable_nodes]
gdf_edges = gdf_edges.loc[ ( gdf_edges['u'].isin(reachable_nodes)) & (gdf_edges['v'].isin(reachable_nodes))]

gdf_nodes.reset_index(inplace=True)

# osmid col contains list, need to convert to single string
for col in gdf_edges.columns:
    gdf_edges.loc[gdf_edges[col].map(lambda v: isinstance(v, list)), col] = gdf_edges.loc[gdf_edges[col].map(lambda v: isinstance(v, list)), col].map(lambda v: "-".join(str(i) for i in v))


gdf_edges.to_file(os.path.join(output_directory, "osmnx_edges.shp"))
gdf_nodes.to_file(os.path.join(output_directory, "osmnx_nodes.shp"))