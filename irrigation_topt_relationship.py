import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from scipy.stats import linregress


def read_and_mask_raster(path, geometry, nodata=np.nan, valid_range=None):
    """
    Read raster data and mask by country boundary.
    """

    with rasterio.open(path) as src:
        data, _ = mask(
            src,
            geometry.geometry,
            crop=False,
            filled=True,
            nodata=nodata
        )

    data = data[0]

    if valid_range is not None:
        data[(data < valid_range[0]) |
             (data > valid_range[1])] = np.nan

    return data


def collect_irrigation_topt_data(
        country_gdf,
        topt_dir,
        irrigation_dir,
        lgrip_file,
        crops,
        n_bins=25):
    """
    Extract irrigation water use and Topt values
    from irrigated cropland pixels and aggregate
    them into irrigation-intensity bins.
    """

    irrigation_all = []
    topt_all = []

    for crop in crops:

        topt_path = os.path.join(
            topt_dir,
            f"Topt_{crop}.tif"
        )

        irrigation_path = os.path.join(
            irrigation_dir,
            f"IWU_{crop}.tif"
        )

        topt = read_and_mask_raster(
            topt_path,
            country_gdf,
            valid_range=(0, 60)
        )

        irrigation = read_and_mask_raster(
            irrigation_path,
            country_gdf,
            valid_range=(0.01, 10000)
        )

        lgrip = read_and_mask_raster(
            lgrip_file,
            country_gdf,
            nodata=-999
        )

        # irrigated cropland mask from LGRIP30
        irrigated_mask = lgrip == 2

        valid = (
            ~np.isnan(topt)
            & ~np.isnan(irrigation)
            & irrigated_mask
        )

        if np.sum(valid) > 0:
            irrigation_all.append(
                irrigation[valid]
            )

            topt_all.append(
                topt[valid]
            )


    if len(irrigation_all) == 0:
        return None, None


    irrigation_all = np.concatenate(
        irrigation_all
    )

    topt_all = np.concatenate(
        topt_all
    )


    # Group pixels into percentile bins
    bins = np.nanpercentile(
        irrigation_all,
        np.linspace(0, 100, n_bins + 1)
    )

    bin_id = np.digitize(
        irrigation_all,
        bins,
        right=False
    )


    irrigation_mean = []
    topt_mean = []


    for i in range(1, n_bins + 1):

        idx = bin_id == i

        if np.sum(idx) > 1:

            irrigation_mean.append(
                np.nanmean(irrigation_all[idx])
            )

            topt_mean.append(
                np.nanmean(topt_all[idx])
            )


    return (
        np.array(irrigation_mean),
        np.array(topt_mean)
    )


def fit_linear_relationship(
        irrigation,
        topt):
    """
    Fit linear relationship between irrigation
    water use and Topt.
    """

    slope, intercept, r_value, p_value, _ = linregress(
        irrigation,
        topt
    )

    return {
        "slope": slope,
        "intercept": intercept,
        "R2": r_value ** 2,
        "p_value": p_value,
        "n_bins": len(irrigation)
    }


def main():

    # Input paths
    shp_path = "China_India.shp"

    topt_dir = "Topt_area"

    irrigation_dir = "irrigation"

    lgrip_file = "LGRIP30_re_mask.tif"


    crops = [
        "maize",
        "rice",
        "soybean",
        "wheat"
    ]


    gdf = gpd.read_file(
        shp_path
    ).to_crs("EPSG:4326")


    for country in ["China", "India"]:

        region = gdf[
            gdf["NAME"].str.contains(
                country,
                case=False
            )
        ]


        irrigation, topt = collect_irrigation_topt_data(
            region,
            topt_dir,
            irrigation_dir,
            lgrip_file,
            crops
        )


        if irrigation is None:
            print(
                f"{country}: no valid data"
            )
            continue


        result = fit_linear_relationship(
            irrigation,
            topt
        )


        print(
            f"\n{country}"
        )

        for key, value in result.items():
            print(
                f"{key}: {value}"
            )


if __name__ == "__main__":
    main()