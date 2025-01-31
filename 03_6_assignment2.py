# Python code for assignment 2
# See assignment2.md for instructions

import pandas as pd
import numpy as np
from plotnine import *

# Read in the data
willowtit = pd.read_csv("data/wtmatrix.csv")
willowtit['status'] = np.where(willowtit['y.1'] == 1, "present", "absent")
willowtit = willowtit[['status','elev']]
willowtit.head()

# Summary (with harmless warnings)
willowtit['bin'] = pd.cut(willowtit['elev'], range(0, 3000, 500))
willowtit.groupby(['bin', 'status']).size()

# Map of switzerland
# nb plotnine does not yet have support for map projections, so this plot is
# spatially distorted but is fine for visualizing the result
swissdem = pd.read_csv("data/switzerland_tidy.csv")

(ggplot()
+ geom_raster(aes(x="x", y="y", fill="Elev_m"), data=swissdem)
+ scale_fill_cmap(name="Elevation (m)")
+ labs(title="Switzerland: DEM")
+ theme_void())
    
# Map of predicted distribution (mockup)
swissdem['pred_status'] = np.where((swissdem['Elev_m'] > 800) & (swissdem['Elev_m'] < 1200), "present", "absent")
dem_present = swissdem.query('pred_status == "present"')

(ggplot()
+ geom_raster(aes(x="x", y="y", fill="Elev_m"), data=swissdem)
+ scale_fill_cmap(name="Elevation (m)")
+ geom_tile(aes(x="x", y="y"), data=dem_present, fill="pink", alpha=0.4)
+ labs(title="Pink: Predicted distribution of Willow Tit in Switzerland")
+ theme_void())
