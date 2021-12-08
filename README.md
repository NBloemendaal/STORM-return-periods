# STORM-return-periods
These scripts are used to calculate the wind speed return periods from the STORM dataset
**IMPORTANT: Please be aware that these scripts are not maintained and NO support is provided!!**

----------------------------------------------------------------------------------------
To calculate the return periods at 10 km resolution, please use (in specified order):
1. holland_model.py + masterprogram.py
2. return_periods_10km_grid.py

----------------------------------------------------------------------------------------
To calculate the return periods at basin-scale and compare them to other extreme-value
distributions, please use: 
1. Basin_wide_return_periods_EV.py

----------------------------------------------------------------------------------------
To calculate the return periods for a selection of 18 coastal cities and 63 islands,
please use:
1. STORM_return_periods_cities_radius.py
1. STORM_return_periods_islands_radius.py +List_of_islands.xlsx
