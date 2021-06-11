# IPython log file

import geopandas as gpd
import pandas as pd
from operator import itemgetter

data_directory = "S:\\CASA_obits_ucfnoth\\1. PhD Work\\GIS Data\\CoventGardenWaterloo\\mastermap-topo_2903032\\mastermap-topo_2903032_0 TopographicArea"
file = 'mastermap TopographicArea.shp'
gdf = gpd.read_file(file)
descr = list(gdf['descriptiv'].unique())


# What descriptiveGroup items fall within the themes I am interested in
gdf.loc[ gdf['theme'].isin(["(1:Roads Tracks And Paths)", 
                            "(2:Land,Roads Track And Paths)", 
                            "(2:Roads Tracks And Paths,Structures)",
                            "(2:Roads Tracks And Paths,Water)"]), 'descriptiv'].unique()


# Select these themes and their associated descriptiveGroups and codes from the attribute data
df_theme = gdf.loc[ gdf['theme'].isin(["(1:Roads Tracks And Paths)", 
                            "(2:Land,Roads Track And Paths)", 
                            "(2:Roads Tracks And Paths,Structures)",
                            "(2:Roads Tracks And Paths,Water)"]), ['theme', 'descriptiv','featureCod']]

df_theme = pd.DataFrame(df_theme)

df_theme.drop_duplicates(inplace=True)

# Selecte the descriptive group items that I want to incorporate into the model

# Unique descriptive groups
descr = list(gdf['descriptiv'].unique())

# This selects all of the same as above apart from includes '(1:General Surface)'
desired_items = itemgetter(1,3,4,5,11,14,19,20)
desired_items(descr)

# Select attribute elements that are associated with these descriptive groups
df_descr = gdf.loc[ gdf['descriptiv'].isin(desired_items(descr)), ['theme', 'descriptiv','featureCod']]

df_descr = pd.DataFrame(df_descr)

df_descr.drop_duplicates(inplace=True)


df_theme.to_csv('topoArea Road Attribute Types.csv')
df_descr.to_csv('topoArea Road Attribute Types descriptiveGroups.csv')