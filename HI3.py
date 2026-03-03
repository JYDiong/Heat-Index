import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from herbie import Herbie

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Create output folders
# --------------------------------------------------
os.makedirs("./image", exist_ok=True)
os.makedirs("./image_rain", exist_ok=True)

# --------------------------------------------------
# Initialization Time (Yesterday 00 UTC)
# --------------------------------------------------
init_date = datetime.utcnow().replace(
    hour=0, minute=0, second=0, microsecond=0
) - timedelta(days=1)

# --------------------------------------------------
# FULL GFS Forecast Hours (0–384)
# 0–120 hourly
# 126–384 every 3 hours
# --------------------------------------------------
forecast_hours = list(range(0, 121)) + list(range(126, 385, 3))

datasets = []

print("Downloading GFS full forecast...")

for fxx in forecast_hours:
    print(f"Downloading F{fxx:03d}")

    H = Herbie(
        date=init_date,
        model="gfs",
        product="pgrb2.0p25",
        fxx=fxx,
    )

    ds = H.xarray(
        ":(TMP:2 m above ground|DPT:2 m above ground|APCP:surface):"
    )

    if isinstance(ds, list):
        ds = xr.merge(ds)

    datasets.append(ds)

# --------------------------------------------------
# Combine all timesteps
# --------------------------------------------------
ds = xr.concat(datasets, dim="time")

print("Total timesteps:", len(ds.time))

# --------------------------------------------------
# Subset Malaysia
# --------------------------------------------------
ds = ds.sel(
    longitude=slice(100, 120),
    latitude=slice(12, 0)
)

# --------------------------------------------------
# Temperature Conversion
# --------------------------------------------------
def kelvin_to_fahrenheit(da):
    return (da - 273.15) * 9/5 + 32

tmp_f = kelvin_to_fahrenheit(ds["t2m"])

# --------------------------------------------------
# Relative Humidity Calculation
# --------------------------------------------------
T = ds["t2m"] - 273.15
Td = ds["d2m"] - 273.15

es = 6.112 * np.exp((17.67 * T) / (T + 243.5))
e  = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))

rh = 100 * (e / es)

# --------------------------------------------------
# Heat Index Calculation
# --------------------------------------------------
def calculate_heat_index(T_f, RH):

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

    return xr.where(T_f <= 80, HI_simple, HI_full)

HI = calculate_heat_index(tmp_f, rh).compute()

# --------------------------------------------------
# Plot Settings
# --------------------------------------------------
bounds = [80, 90, 103, 124]
colors = ['#ffe082', '#ffb74d', '#e64a19']
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

lon = HI.longitude
lat = HI.latitude
times = HI.time
n_times = len(times)

init_time_str = init_date.strftime("%Y-%m-%d %H:%M UTC")

# --------------------------------------------------
# Plot Heat Index Maps
# --------------------------------------------------
print("Generating Heat Index maps...")

for i in range(n_times):

    valid_time = pd.to_datetime(times[i].values)
    valid_time_str = valid_time.strftime("%Y-%m-%d %H:%M UTC")

    field = HI[i].values
    field = np.where(field >= 80, field, np.nan)

    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.contourf(
        lon, lat, field,
        levels=bounds,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        extend="both"
    )

    ax.coastlines()

    ax.set_title(
        f"Forecast Heat Index\nInit: {init_time_str}\nValid: {valid_time_str}",
        fontsize=9
    )

    cbar = plt.colorbar(
        im, orientation="horizontal",
        pad=0.02, aspect=30, shrink=0.8
    )

    cbar.set_ticks([85, 96.5, 113.5])
    cbar.set_ticklabels([
        "80–90: Caution",
        "90–103: Ext. Caution",
        "103–124: Danger"
    ])
    cbar.ax.tick_params(labelsize=6)

    plt.savefig(f"./image/heat_index_{i:03d}.png", dpi=300)
    plt.close()

# --------------------------------------------------
# 3-Hourly Rainfall Maps
# --------------------------------------------------
print("Generating rainfall maps...")

rain = ds["tp"]
rain_3hr = rain.diff("time").compute()

for i in range(len(rain_3hr.time)):

    valid_time = pd.to_datetime(rain_3hr.time[i].values)
    valid_time_str = valid_time.strftime("%Y-%m-%d %H:%M UTC")

    field = rain_3hr[i].values

    plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())

    im = ax.contourf(
        lon, lat, field,
        levels=np.arange(0, 50, 5),
        cmap="Blues",
        transform=ccrs.PlateCarree(),
        extend="max"
    )

    ax.coastlines()

    ax.set_title(
        f"3-Hourly Accumulated Rainfall (mm)\nInit: {init_time_str}\nValid: {valid_time_str}",
        fontsize=9
    )

    cbar = plt.colorbar(
        im, orientation="horizontal",
        pad=0.02, aspect=30, shrink=0.8
    )
    cbar.set_label("Rainfall (mm)", fontsize=8)

    plt.savefig(f"./image_rain/rainfall_{i:03d}.png", dpi=300)
    plt.close()
