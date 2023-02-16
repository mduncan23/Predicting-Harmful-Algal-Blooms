# Predicting Harmful Algal Blooms Using Satellite and Climate Data

**Author**: [Matthew Duncan](mailto:mduncan0923@gmail.com)

![lake](/Images/lake.jpg)

## Overview
Cyanobacteria, a harmful algal bloom (HAB), can have a severe impact on water quality and cause significant harm to the ecosystem, humans, pets, and marine life. This analysis aims to demonstrate that Cyanobacteria severity levels can be predicted with minimal error by utilizing data already collected by government agencies. I have utilized data on the recorded severity levels of HABs from the Environmental Protection Agency (EPA), satellite imagery and elevation data from the National Aeronautics and Space Administration (NASA), and climate data from the National Oceanic and Atmospheric Administration (NOAA). 

The focus of this analysis was to minimize the error in the predictions rather than to achieve a high level of accuracy. The model is able to predict with a Regional root mean squared error (RMSE) of less than .75 on a severity scale of 1-5.




![satellite](/Images/satellite.jpg)

## Business Understanding

The EPA, NASA, and NOAA have come together in a collaborative effort to predict the presence of Cyanobacteria in freshwater sources.

[Cyanobacteria](https://www.epa.gov/cyanohabs), a harmful algal bloom (HAB), can have a severe impact on water quality and cause significant harm to the ecosystem, humans, pets, and marine life. HABs have the potential to:

- Disrupt ecosystems by hindering sunlight and oxygen
- Produce toxins that are hazardous to human and animal health
- Transfer these toxins to other aquatic creatures, posing a threat to aquaculture and agriculture

This analysis aims to demonstrate that Cyanobacteria severity levels can be predicted with minimal error by utilizing data already collected by government agencies. If successful, this proof of concept will secure further funding to make the predictive model accessible to local governments, helping to protect citizens and the environment. Furthermore, if successful, this model would also reduce costs and manpower in the long term for HAB screening and monitoring.

### Cost of Errors

Incorrectly predicting the presence and severity of Harmful Algal Blooms (HABs) can have negative effects on the ecosystem and the community. If the blooms are not accurately predicted, there is a risk of underestimating the potential harm they can cause. This can lead to a lack of preparation and response to the blooms, allowing them to persist and worsen.

Underestimating the impact of HABs can result in the continued exposure of humans, pets, and marine life to toxic substances, causing harm to their health and potentially leading to fatalities. Additionally, industries such as aquaculture and agriculture can be severely impacted by the persistence of HABs, leading to decreased productivity and potential financial losses.

Accurate predictions of HABs are crucial in order to ensure the safety of the community and the preservation of the ecosystem. Incorrect predictions can result in a lack of proper response and action, allowing HABs to cause harm and destabilize the environment.

![algea](/Images/algea.jpg)

## Understanding the Data

Data is available from all agencies, the EPA has manually sampled over 23,500 sites across the United States from 2013 through 2021 and provided severity levels based on Cyanobacteria count for over 17,000 sites. NASA has provided access to their databank of satellite imagery. NOAA has provided access to their historical climate database.

HABs typically form within a matter of days to weeks. For this analysis, I will be using data within a 15 day range prior to EPA sampling. 

**Note about satelitte and climate data:** Due to the instability of the API and the time frame required to pull data for all EPA samples, the data was accumulated in batches. A sample pull of the data is located in the Final Notebook, though more insight into the methods behind gathering the satellite data and climate data can be found in notebooks: "Batch_pulling_data" and "Climate_data_pulls", respectively. 

In this repo, I have included the `hab_functions.py` with custom functions needed to run this notebook. I have also the `environment.yml` to ensure that you have the correct environment to run the notebook.

### EPA Data


#### Seasonal Sampling
The first sample was taken January 4th, 2013 and the last sample was taken December 29th, 2021. The majority (roughly 45%) of sampling took place in summer months. Most winter samples were taken along the coast, particularly California. Spring samples start to show more sampling in the Midwest and summer and autumn sampling are spread throughout the country.

![sample_seasons](/Images/seasonal_samples.jpg)



#### Severity by Location
Recorded data on the severity of HABs across the continental United States ranges from 1-5 based on bacteria count with the majority (44%) of severity levels being 1. Mapping the severity of the algal blooms by location lets us see that California is impacted by some of the most consistently high levels of severity while the Great Lakes region tends to have lower levels of severity.

![severity_by_location](/Images/severity_by_location.jpg)



#### Regional Trends Over Time
When looking at regional trends over time, there's definitely cyclical patterns in the data. The Southern Region has the most clearly defined seasonality. The data here matches with what we would expect from the severity plot above, with the west coast having consistently higher levels of severity.

![regional_over_time](/Images/regional_over_time.jpg)

### NASA Satellite Imagery Data
The majority of the data for this analysis will be coming from satellite images captured by the Sentinel and LandSat satellites. The LandSat program (a joint NASA/USGS program) provides the longest continuous space-based record of Earth's land.

I'll be using a range of 31 miles around the sample site to pull potential satellite images then ensuring that each image actually contains the specified site coordinates. Imagery data is taken from a 100-meter bounding box surrounding the exact sample location.

#### Large Image
I first take a larger satellite image like the one below to confirm that the coordinates align with a body of water. This image shows a lake with level 5 severity and you can visually see green swirls in the water representing the HAB.

![first_sat](/Images/first_sat.jpg)

#### Zoomed Image
This image is now zoomed in to the eastern coastline to get a better idea of where the EPA sample site was located. This analysis utilizes one more layer of zooming in, though the images will not be shown here.

![second_sat](/Images/second_sat.jpg)

### Satellite Elevation Data
I have also pulled in elevation data from Planetary Computer through the Copernicus Digital Elevation Model. This model uses resolution up to 30 meters and measures elevation above sea level. I will be using the same 100 m buffer around the sample site and taking an average elevation for the specific location.

### NOAA Climate Data
NOAA's High-Resolution Rapid Refresh (HRRR) dataset is a real-time and historical atmospheric modeling system. This analysis used temperature data taken at 12pm CST for one day and 5 day periods prior to the EPA sampling event. This was done to show how temperature may have influenced the HAB prior to sampling.

## Modeling
This analysis utilized a custom scoring method: each Region's RMSE was calculated and then averaged. The dummy model, utilizing the most frequent severity as it's strategy, received a Regional RMSE score of 1.65. 

I utilized an iterative approach to find the best model for predicting the severity of harmful algal blooms. I attempted several different modeling algorithms to find the best blend of cross-validated scores on training and test data in an attempt to maximize scores while reducing overfitting. I then compared the cross validated results to a seperate validation set. 

After comparing Logistic Regression, Random Forest, XGBoost, LinearSVC, Multi-layer Perceptron (MLP), K-Nearest Neighbors, and Stacking Classifiers, the Stacking Classifier provided the best results. The chosen Stacking Classifier model performs the best of all models, scoring .738 on unseen validation data and returning cross validated results of .776 for test data and .413 on training data. While it does not have the lowest validation score of all data, it does have the best combined results of all three indicators (unseen data, cross validated data, and reducing overfitting). This model is still overfit though.

### Visualizing Errors
My model predicted the correct severity level over 2/3rds of the time on held-aside validation data. For sample sites that were incorrectly predicted, 71% were only off in their predictions by one point and 27% were only off in there predictions by two points. Only 2% of predictions incorrectly predicted by more than 2 points. The majority of errors were located in the Southern Region of the United States. Below are geographical locations of all sample sites that were incorrectly predicted.

![mapped_errors](/Images/mapped_errors.jpg)

## Conclusions

### Analysis
This analysis has run dozens of models to predict the severity of harmful algal blooms in inland bodies of water. Ultimately the best model was the Stacking Classifier using Logistic Regression, K-Nearest Neighbors, Random Forest, and XGBoost. The 5-fold cross validated score was .7768 for the Regional Root Mean Squared Error. When Tested on unseen validation data, the model scored:

- RMSE for south (n=2744): 0.8968
- RMSE for midwest (n=626): 0.824
- RMSE for west (n=1030): 0.399
- RMSE for northeast (n=309): 0.8341
- Final score: 0.7384900178900713

The model has a much better predictive ability for the western region than the other regions. This makes sense when we look at the EPA data and can visually see that the western region has a relatively consistent severity average of between about 3.5 and 4. Visually, the Northeast and Midwest appear to have the most variability in their severity, though the model is better able to predict for these regions than for the south.

It is important to note that the safety of people, animals, and the environment is at stake, so the focus of this analysis was to minimize the error in the predictions rather than to achieve a high level of accuracy. The use of root mean squared error was appropriate for this purpose as it measures the deviation between the predicted values and the actual values, and the goal was to keep this deviation as low as possible.

This analysis has proven that machine learning using existing EPA sample data, NASA Satellite Imagery, and NOAA climate data can be used to better predict the severity of harmful algal blooms in inland bodies of water.

Additional Funding to continue research and implement improvements on the model is vital to the safety and security of our aquatic environments. In the future considerations section below, I highlight some of the areas where additional research could benefit the model.

### Future Considerations
There are several areas for improvement that could further enhance the performance of the predictive model. These include:

- Incorporating multiple satellite images for each site: Utilizing multiple satellite images for each site has the potential to significantly improve the model's predictive ability by providing more data for the model to learn from.


- Using more climate data: In this analysis, only two points of climate data (temperature from 1 and 5 days prior to EPA sampling) were used. Incorporating more climate data points, such as wind gusts and precipitation, from the NOAA HRRR system would greatly improve the model's ability to predict HABs.


- Using more temperature data: In this analysis, only one temperature point was taken for each day. Utilizing hourly temperature data from the NOAA HRRR system would significantly improve the model's ability to predict HABs.

Additional funding and resources for continued research and implementation of these improvements are crucial for ensuring the safety and protection of the environment and the communities that rely on freshwater sources.

## For More Information

See the full analysis in [Jupyter Notebook Introduction](./Introduction_and_Data.ipynb) and [Jupyter Notebook Conclusion](./Modeling_and_Conclusion.ipynb) or review this [presentation](./presentation.pdf).

**For additional info, contact:**
- Matthew Duncan: mduncan0923@gmail.com

![closing_image](/Images/closing_image.jpg)

## Repository Structure

```
├── Data
│   ├── clean_test_data.csv
│   ├── clean_train_data.csv
│   ├── metadata.csv
│   ├── train_labels.csv
├── Images
│   ├── algea.jpg
│   ├── closing_image.jpg
│   ├── first_sat.jpg
│   ├── lake.jpg
│   ├── mapped_errors.jpg
│   ├── regional_over_time.jpg
│   ├── satellite.jpg
│   ├── seasonal_samples.jpg
│   ├── second_sat.jpg
│   ├── severity_by_location.jpg
│   Scratch_Notebooks
│   ├── Batch_pulling_data.ipynb
│   ├── Climate_data_pulls.ipynb
|   └── hab_functions.py
├── environment.yml
├── Introduction_and_Data.ipynb
├── Modeling_and_Conclusion.ipynb
├── hab_functions.py
├── presentation.pdf
└── README.md
```
