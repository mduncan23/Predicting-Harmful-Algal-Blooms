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