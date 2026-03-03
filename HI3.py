import pandas as pd
import xarray as xr
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from herbie import Herbie

# Create folders
os.makedirs('./image_rain', exist_ok=True)
os.makedirs('./image', exist_ok=True)

# --------------------------------------------------
# Date (Yesterday 00 UTC)
# --------------------------------------------------
date = datetime.utcnow().replace(
    hour=0, minute=0, second=0, microsecond=0
) - timedelta(days=1)

datasets = []

for fxx in range(0, 25):   # 0 to 24 forecast hours
    H = Herbie(
        date=date,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx
    )

    ds = H.xarray(
        ":(TMP:2 m above ground|DPT:2 m above ground|APCP:surface):"
    )

    if isinstance(ds, list):
        ds = xr.merge(ds)

    datasets.append(ds)

# Combine all forecast hours into one dataset
ds = xr.concat(datasets, dim="time")
# --------------------------------------------------
# Subset Malaysia
# --------------------------------------------------
ds_Malaysian = ds.sel(
    longitude=slice(100, 120),
    latitude=slice(12, 0)
)

# --------------------------------------------------
# Temperature Conversion
# --------------------------------------------------
def temp2F(da):
    celsius = da - 273.15
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit


tmp_2m_f = temp2F(ds_Malaysian["t2m"])

# --------------------------------------------------
# Compute Relative Humidity from Temp + Dewpoint
# --------------------------------------------------
T = ds_Malaysian["t2m"] - 273.15

if "d2m" not in ds_Malaysian:
    raise ValueError("Dewpoint (d2m) not found in dataset")

Td = ds_Malaysian["d2m"] - 273.15

es = 6.112 * xr.ufuncs.exp((17.67 * T) / (T + 243.5))
e  = 6.112 * xr.ufuncs.exp((17.67 * Td) / (Td + 243.5))

rh2m = 100 * (e / es)

# Store into dataset
ds_Malaysian["rh2m"] = rh2m

# --------------------------------------------------
# Heat Index Calculation
# --------------------------------------------------
def calculate_heat_index_combined(T_f, RH):

    HI_simple = 0.5 * (
        T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094)
    )

    HI_full = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * RH
        - 0.22475541 * T_f * RH
        - 6.83783e-3 * T_f**2
        - 5.481717e-2 * RH**2
        + 1.22874e-3 * T_f**2 * RH
        + 8.5282e-4 * T_f * RH**2
        - 1.99e-6 * T_f**2 * RH**2
    )

    HI = xr.where(T_f <= 80, HI_simple, HI_full)

    return HI


HI = calculate_heat_index_combined(tmp_2m_f, rh2m)

HI_computed = HI.compute()

import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.colors import BoundaryNorm
import numpy as np
import matplotlib.colors as mcolors

# Define boundaries and labels
bounds = [80, 90, 103, 124]
labels = ["Caution", "Ext. Caution", "Danger"]
colors = ['#ffe082', '#ffb74d', '#e64a19']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

longitude = HI_computed.longitude
latitude = HI_computed.latitude
data = HI_computed.values
timestep = HI_computed.time

# Plot Heat Index
for idx in range(41):
    field = data[idx]
    forecast_time = timestep[idx]
    formatted_time = pd.to_datetime(forecast_time.values).strftime('%Y-%m-%d %H:%M')
    masked_field = np.where(field >= 80, field, np.nan)  # mask below 80°F

    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(longitude, latitude, masked_field, levels=bounds,
                     transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, extend='both')
    ax.coastlines()
    ax.set_title(f"Forecast Heat Index, init: {str(adate)}, \n valid: {str(formatted_time)}", fontsize=10)
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.02, aspect=30, shrink=0.8, ax=ax, location='bottom')
    cbar.set_label("HI Risk Level", fontsize=8)
    cbar.set_ticks([85, 96.5, 113.5])
    cbar.set_ticklabels(["80–90: Caution", "90–103: Ext. Caution", "103–124: Danger"])
    cbar.ax.tick_params(labelsize=6)
    plt.savefig(f'./image/Heat_index_init_{idx}.png', dpi=300)
    plt.close()

# Rainfall calculation
rain = ds_Malaysian['apcpsfc']  # kg m-2 == mm
rain_3hr = rain.diff(dim='time')
rain_3hr_computed = rain_3hr.compute()

for idx in range(41):
    field = rain_3hr_computed[idx]
    forecast_time = timestep[idx]
    formatted_time = pd.to_datetime(forecast_time.values).strftime('%Y-%m-%d %H:%M')

    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(longitude, latitude, field, transform=ccrs.PlateCarree(),
                     cmap='Blues', levels=np.arange(0, 50, 5), extend='max')
    ax.coastlines()
    ax.set_title(f"3-hourly Accumulated Rainfall (mm), init: {adate}, \n valid: {formatted_time}", fontsize=10)
    cbar = plt.colorbar(im, orientation='horizontal', pad=0.02, aspect=30, shrink=0.8, ax=ax, location='bottom')
    cbar.set_label("3-hourly Accumulated Rainfall (mm)", fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    plt.savefig(f'./image_rain/Rainfall_map_{idx}.png', dpi=300)
    plt.close()
