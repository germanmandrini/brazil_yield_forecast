# In-Season Yield Forecast
In-Season Yield prediction using crop modeling, remote sensing, and machine learning
Crop and Region of Interest: Soybean for Mato Grosso, Brazil.

## Data Sources:
- Historical Yield, planted area and production (2011-2022)
- Planting Progress (2015-2023)
- Harvest Progress (2018-2022)
- Weather (2000-2023)
- NDVI (2013-2023)
- Soybean Area 2022 (raster) 
- Regions (shp)
  
## Methodology:
Used APSIM Next Generation and the R package APSIMX for running simulations
The analysis was done using python. The output of the crop model was combined with ground truth data (weather, remote sensing, planting date) to train a Catboost Machine Learning Model.
In-season predictions for yield and harvest progess curve were obtained.

Presentation: [Slides](https://docs.google.com/presentation/d/10EecWW7bXi1HvKpNrysvZ3Zqz_XWRexhB-QB0XmXSlo/edit#slide=id.g1ef36b1199d_0_93)
