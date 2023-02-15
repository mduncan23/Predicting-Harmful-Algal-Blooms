from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

import geopandas as gpd
from shapely.geometry import Point
import geopy.distance as distance

import pandas as pd
import numpy as np

import planetary_computer as pc
from pystac_client import Client

from datetime import datetime
from datetime import timedelta

import rioxarray
import odc.stac

############################

class MinMaxScaler3D(MinMaxScaler):

    def fit_transform(self, X, y=None):
        x = np.reshape(X, newshape=(X.shape[0]*X.shape[1], X.shape[2]))
        return np.reshape(super().fit_transform(x, y=y), newshape=X.shape)
    
 ############################

def create_bbox(coords_df, dist=5):

    '''
    Input a dataframe that includes lat/long and outputs a list of bbox's for each row 
    '''
    bbox_list = []
    
    for n in range(len(coords_df)):
        coord = coords_df.iloc[n]
        bbox = []
        bbox.append(distance.distance(miles=dist).destination((coord.latitude, coord.longitude), bearing=270)[1])
        bbox.append(distance.distance(miles=dist).destination((coord.latitude, coord.longitude), bearing=180)[0])
        bbox.append(distance.distance(miles=dist).destination((coord.latitude, coord.longitude), bearing=90)[1])
        bbox.append(distance.distance(miles=dist).destination((coord.latitude, coord.longitude), bearing=0)[0])
        bbox_list.append(bbox)
    coords_df[f"bbox"] = bbox_list
    return coords_df

############################

def create_big_crop_bbox(coords_df, dist=3000):

    '''
    Input a dataframe that includes lat/long and outputs a list of bbox's for each row 
    '''
    bbox_list = []
    
    for n in range(len(coords_df)):
        coord = coords_df.iloc[n]
        bbox = []
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=270)[1])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=180)[0])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=90)[1])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=0)[0])
        bbox_list.append(bbox)
    coords_df[f"big_crop_bbox"] = bbox_list
    return coords_df

############################

def create_small_crop_bbox(coords_df, dist=500):

    '''
    Input a dataframe that includes lat/long and outputs a list of bbox's for each row 
    '''
    bbox_list = []
    
    for n in range(len(coords_df)):
        coord = coords_df.iloc[n]
        bbox = []
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=270)[1])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=180)[0])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=90)[1])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=0)[0])
        bbox_list.append(bbox)
    coords_df[f"small_crop_bbox"] = bbox_list
    return coords_df

############################

def create_tiny_crop_bbox(coords_df, dist=100):

    '''
    Input a dataframe that includes lat/long and outputs a list of bbox's for each row 
    '''
    bbox_list = []
    
    for n in range(len(coords_df)):
        coord = coords_df.iloc[n]
        bbox = []
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=270)[1])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=180)[0])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=90)[1])
        bbox.append(distance.distance(meters=dist).destination((coord.latitude, coord.longitude), bearing=0)[0])
        bbox_list.append(bbox)
    coords_df[f"tiny_crop_bbox"] = bbox_list
    return coords_df

############################

def get_date_range(date_df, range_in_days=15):
    
    '''
    input a dataframe with a date as the index and specify range of dates.
    Will create new column in df that has daterange for sat image pulling.
    '''
    
    range_list = []
    
    for n in date_df['date']:
        date = pd.to_datetime(n).strftime(format="%Y-%m-%d")
        day_range = (n-timedelta(range_in_days)).strftime(format="%Y-%m-%d")
        final_range = f"{day_range}/{date}"
        
        
        range_list.append(final_range)
    date_df[f"date_range"] = range_list
    return date_df

############################

def get_important_info(df, dist=5, range_in_days=15, big_crop_dist=1000, small_crop_dist= 500, tiny_crop_dist=100):
    
    '''
    input a dataframe to get a bounding box in specified # miles (dist).
    input a dataframe with date set as the index to output a new column with date/time  for the day range
    '''
    
    get_date_range(df, range_in_days=range_in_days)
    create_bbox(df, dist=dist)
    create_big_crop_bbox(df, dist=big_crop_dist)
    create_small_crop_bbox(df, dist=small_crop_dist)
    create_tiny_crop_bbox(df, dist=tiny_crop_dist)
    return(df)

############################

# Can use this if I decide to use multiple satelitte images
def get_sat_info(df):

    '''
    input a dataframe and get a dictionary with satellite information for each row in the dataframe
    '''
    
    catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    
    sat_dict = {}
    for index in range(len(df)):
        row = df.iloc[index]

        # Get all satellite images
        search = catalog.search(collections=["sentinel-2-l2a", "landsat-c2-l2"],
                                bbox=row['bbox'],
                                datetime=row['date_range'],
                                query={'eo:cloud_cover': {'lt':100}}
    )


        # Going through Satellite info

        # search for sat images and create a dataframe with results for one sample
        search_items = [item for item in search.item_collection()]


        pic_details = []
        for pic in search_items:
            pic_details.append(
            {
            'item': pic,
            'satelite_name':pic.collection_id,
            'img_date':pic.datetime.date(),
            'cloud_cover(%)': pic.properties['eo:cloud_cover'],
            'img_bbox': pic.bbox,
            'min_long': pic.bbox[0],
            "max_long": pic.bbox[2],
            "min_lat": pic.bbox[1],
            "max_lat": pic.bbox[3]
            }
            )

        temp_df = pd.DataFrame(pic_details)

        # Check to make sure sample location is actually within sat image
        temp_df['has_sample_point'] = (
            (temp_df.min_lat < row.latitude)
            & (temp_df.max_lat > row.latitude)
            & (temp_df.min_long < row.longitude)
            & (temp_df.max_long > row.longitude)
        )

        temp_df = temp_df[temp_df['has_sample_point'] == True]
        sat_dict[row['uid']] = temp_df
        
    return sat_dict

############################

# Can use this if I decide to use multiple satelitte images
def try_get_sat_info(df):

    '''
    input a dataframe and get a dictionary with satellite information for each row in the dataframe
    '''
    
    catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    
    sat_dict = {}
    for index in range(len(df)):
        row = df.iloc[index]

        # Get all satellite images
        search = catalog.search(collections=["sentinel-2-l2a", "landsat-c2-l2"],
                                bbox=row['bbox'],
                                datetime=row['date_range'],
                                query={'eo:cloud_cover': {'lt':100}}
    )


        # Going through Satellite info

        # search for sat images and create a dataframe with results for one sample
        search_items = [item for item in search.item_collection()]


        pic_details = []
        for pic in search_items:
            pic_details.append(
            {
            'item': pic,
            'satelite_name':pic.collection_id,
            'img_date':pic.datetime.date(),
            'cloud_cover(%)': pic.properties['eo:cloud_cover'],
            'img_bbox': pic.bbox,
            'min_long': pic.bbox[0],
            "max_long": pic.bbox[2],
            "min_lat": pic.bbox[1],
            "max_lat": pic.bbox[3]
            }
            )

        temp_df = pd.DataFrame(pic_details)
        try:
            # Check to make sure sample location is actually within sat image
            temp_df['has_sample_point'] = (
                (temp_df.min_lat < row.latitude)
                & (temp_df.max_lat > row.latitude)
                & (temp_df.min_long < row.longitude)
                & (temp_df.max_long > row.longitude)
            )

            temp_df = temp_df[temp_df['has_sample_point'] == True]
            sat_dict[row['uid']] = temp_df
        except:
            sat_dict[row['uid']] = temp_df
        
    return sat_dict

############################


def pick_best_sat(df, sat_dict):
    
    '''
    input a dataframe and dictionary of satellite images and returns a dataframe with the best satellite image
    '''
    
    # picking the best
    # inputs would need to be df and dictionary
    best_sat_df = pd.DataFrame()
    row_count=0
    invalid_sats = 0
    for index in range(len(df)):
        row = df.iloc[index]

        name = row['uid']
        temp_df = sat_dict[name]
        temp_df = temp_df.reset_index()
        # checking to see if there's only one image and adding it to df if so
        if len(temp_df) == 1:
            temp_df = temp_df.reset_index().drop(['index','min_long', 'max_long', 'min_lat', 'max_lat'], axis=1)
            row = pd.DataFrame(row).T.reset_index().join(temp_df, how='outer')
            row = row.set_index(pd.Series(row_count)).drop(['level_0', 'index'], axis=1)
            best_sat_df = pd.concat([best_sat_df, row])
            row_count+=1

        # checking if no images
        elif len(temp_df) == 0:
            invalid_sats +=1
            row = pd.DataFrame(row).T.reset_index()
            row = row.set_index(pd.Series(row_count)).drop('index', axis=1)
            best_sat_df = pd.concat([best_sat_df, row])
            row_count+=1
            continue

        # There are many satellite images, need to narrow it down
        else:
            # first checking for any sentinel satelites
            if len(temp_df[temp_df['satelite_name'].str.contains('entinel')]) >0:
                    temp_df = temp_df[temp_df['satelite_name'].str.contains('entinel')]

                    # if only one sentinel, add to df and move on
                    if len(temp_df) == 1:
                        temp_df = temp_df.reset_index().drop(['index','min_long', 'max_long', 'min_lat', 'max_lat'], axis=1)
                        row = pd.DataFrame(row).T.reset_index().join(temp_df, how='outer')
                        row = row.set_index(pd.Series(row_count)).drop(['level_0', 'index'], axis=1)
                        best_sat_df = pd.concat([best_sat_df, row])
                        row_count+=1
                    # if many sentinel, check for images with low cloud cover
                    else:
                        # checking for clouds less than 30%
                        if len(temp_df[temp_df['cloud_cover(%)'] <= 30]) >0:
                            temp_df = temp_df[temp_df['cloud_cover(%)'] <= 30]

                            # add the row with the closest date
                            temp_df = temp_df.sort_values('img_date', ascending=False).reset_index().drop(['index','min_long', 'max_long', 'min_lat', 'max_lat'], axis=1)
                            temp_df = pd.DataFrame(temp_df.loc[0]).T
                            row = pd.DataFrame(row).T.reset_index().join(temp_df, how='outer')
                            row = row.set_index(pd.Series(row_count)).drop(['level_0', 'index'], axis=1)
                            best_sat_df = pd.concat([best_sat_df, row])
                            row_count+=1
                        else:
                            # If there's only images with a clouds over 30%, 
                            # pick the one with the least clouds
#                             print('\t\tvery cloudy sentinel')
                            temp_df = temp_df.sort_values('cloud_cover(%)', ascending=True).reset_index().drop(['index','min_long', 'max_long', 'min_lat', 'max_lat'], axis=1)
                            temp_df = pd.DataFrame(temp_df.loc[0]).T
                            row = pd.DataFrame(row).T.reset_index().join(temp_df, how='outer')
                            row = row.set_index(pd.Series(row_count)).drop(['level_0', 'index'], axis=1)
                            best_sat_df = pd.concat([best_sat_df, row])
                            row_count+=1

            else:
                if len(temp_df[temp_df['cloud_cover(%)'] <= 30]) >0:
                    temp_df = temp_df[temp_df['cloud_cover(%)'] <= 30]

                    # add the row with the closest date
                    temp_df = temp_df.sort_values('img_date', ascending=False).reset_index().drop(['index','min_long', 'max_long', 'min_lat', 'max_lat'], axis=1)
                    temp_df = pd.DataFrame(temp_df.loc[0]).T
                    row = pd.DataFrame(row).T.reset_index().join(temp_df, how='outer')
                    row = row.set_index(pd.Series(row_count)).drop(['level_0', 'index'], axis=1)
                    best_sat_df = pd.concat([best_sat_df, row])
                    row_count+=1
                else:
                    # If there's only images with a clouds over 30%, 
                    # pick the one with the least clouds
                    temp_df = temp_df.sort_values('cloud_cover(%)', ascending=True).reset_index().drop(['index','min_long', 'max_long', 'min_lat', 'max_lat'], axis=1)
                    temp_df = pd.DataFrame(temp_df.loc[0]).T
                    row = pd.DataFrame(row).T.reset_index().join(temp_df, how='outer')
                    row = row.set_index(pd.Series(row_count)).drop(['level_0', 'index'], axis=1)
                    best_sat_df = pd.concat([best_sat_df, row])
                    row_count+=1



    print(f'{len(df)} attempts. {invalid_sats} failures.')
    return best_sat_df

############################

def get_arrays_from_sats(df):
    
    
    '''
    input a dataframe with satellites in it and get a dictionary with arrays 
    that came from cropped images around the sample area
    '''

# Now to get images from the satellites
    array_dict = {}
    scaler = MinMaxScaler3D(feature_range=(0,255))
    error_count = 0
    attempt_count = 0
    for index in range(len(df)):
        row = df.iloc[index]


        try:
            attempt_count +=1
        # checking to see which satellite it came from
            if 'sentinel' in row['satelite_name']:
                # Setting tiny crop box for image
                minx, miny, maxx, maxy = row['tiny_crop_bbox']
                # getting the image
                image = rioxarray.open_rasterio(pc.sign(row['item'].assets["visual"].href)).rio.clip_box(
                        minx=minx,
                        miny=miny,
                        maxx=maxx,
                        maxy=maxy,
                        crs="EPSG:4326",
                    )

                image_array = image.to_numpy()
                img_array_trans = np.transpose(image_array, axes=[1, 2, 0])
                # storing array of image in dictionary
                array_dict[row['uid']] = img_array_trans

            else:
                # getting the image from the LandSat satellite
                minx, miny, maxx, maxy = row['tiny_crop_bbox']
                image = odc.stac.stac_load(
                        [pc.sign(row['item'])], bands=["red", "green", "blue"], bbox=[minx, miny, maxx, maxy]
                    ).isel(time=0)

                image_array = image[["red", "green", "blue"]].to_array()
                img_array_trans = np.transpose(image_array.to_numpy(), axes=[1, 2, 0])
                # scaling the image so its the same scale as the sentinel ones
                scaled_img = scaler.fit_transform(img_array_trans)
                # storing array of image in dictionary
                array_dict[row['uid']] = scaled_img
                

        except:
            error_count +=1
            
            
    print(f'{attempt_count} attempted. {error_count} failures.')
    return array_dict

############################

def get_features(df, img_arrays):
    '''
    input a dataframe and a list of integers and create features from arrays
    '''
    feature_df = pd.DataFrame()
    for index in range(len(img_arrays.keys())):
        feature_dict = {}
        key =list(img_arrays.keys())[index]
        temp_array = img_arrays[key]
        for n, color in enumerate(['red', 'green', 'blue']):
            feature_dict['uid'] = key
            feature_dict[f'{color}_mean'] = np.mean(temp_array[:,:,n])
            feature_dict[f'{color}_median'] = np.median(temp_array[:,:,n])
            feature_dict[f'{color}_max'] = np.max(temp_array[:,:,n])
            feature_dict[f'{color}_min'] = np.min(temp_array[:,:,n])
            feature_dict[f'{color}_sum'] = np.sum(temp_array[:,:,n])
            feature_dict[f'{color}_product'] = np.prod(temp_array[:,:,n])
        feature_df = pd.concat([feature_df, pd.DataFrame(feature_dict, index=[index])], )

    feature_df = df.merge(feature_df, how='outer', on='uid')
    return feature_df

############################

# A function to get it all in one
def get_sat_to_features(df):
    
    '''
    input a dataframe of raw data and get sat images, convert to arrays, and turn into features.
    '''
    catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    
    # get sat info
    satelite_dict = get_sat_info(df)
    
    # pick best sat
    single_df = pick_best_sat(df, satelite_dict)
    
    # get image arrays from best sats
    img_arrays = get_arrays_from_sats(single_df)
    
    # get a dataframe with relevant features
    feature_df = get_features(single_df, img_arrays)
    
    return feature_df

############################

# A function to try get it all in one
def try_get_sat_to_features(df):
    
    '''
    input a dataframe of raw data and get sat images, convert to arrays, and turn into features.
    '''
    catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
    )
    
    # get sat info
    satelite_dict = try_get_sat_info(df)
    
    # pick best sat
    single_df = pick_best_sat(df, satelite_dict)
    
    # get image arrays from best sats
    img_arrays = get_arrays_from_sats(single_df)
    
    # get a dataframe with relevant features
    feature_df = get_features(single_df, img_arrays)
    
    return feature_df


############################

# not used in the analysis but good to have
def clean_data(df):
    '''
    input dataframe with all data and clean it.
    '''
    # only keeping cols that I need
    model_df = df[['date', 'latitude', 'longitude', 'season', 'img_date',
            'red_mean', 'red_median', 'red_max', 'red_min','red_sum',
            'red_product', 'green_mean', 'green_median', 'green_max',
            'green_min', 'green_sum', 'green_product', 'blue_mean',
            'blue_median','blue_max', 'blue_min', 'blue_sum', 'blue_product', 'severity']]
    # dropping nulls
    model_df = model_df.dropna()
    # converting to correct type
    model_df['date'] = model_df['date'].apply(lambda x: datetime.date(x))
    # getting difference from image date to sample date and creating feature
    model_df['days_from_sat_to_sample'] = model_df['date'] - model_df['img_date']
    # converting to int
    model_df['days_from_sat_to_sample'] = model_df['days_from_sat_to_sample'].dt.days
    # converting from datetime to an int
    model_df['date'] = model_df['date'].apply(lambda x: x.toordinal())
    model_df['img_date'] = model_df['img_date'].apply(lambda x: x.toordinal())
    # converting from string to float
    model_df['latitude'] = model_df['latitude'].apply(lambda x: x.astype(float))
    model_df['longitude'] = model_df['longitude'].apply(lambda x: x.astype(float))
    
    # One hot encoding seasons
    ohe = OneHotEncoder(sparse=False)
    seasons = ohe.fit_transform(model_df[['season']])
    cols = ohe.get_feature_names_out()
    # converting new ohe to dataframe
    seasons = pd.DataFrame(seasons, columns=cols, index=model_df.index)
    model_ohe = pd.concat([model_df.drop('season', axis=1), seasons], axis=1)
    return model_ohe