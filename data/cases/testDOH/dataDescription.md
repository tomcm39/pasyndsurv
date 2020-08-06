# Model: PA DOH (test data)

## Raw Data Link: 
N/A

## Database Description:
This is data directly from the PA DOH website. 

## Database Link:
https://www.health.pa.gov/topics/disease/coronavirus/Pages/Archives.aspx

## How to Use

python3 scrapData.py

This script will scrape the data off the DOH archive, create files for each day, and then read and write to those files with calculated testing data.

Open the data directory to view the downloaded files. 

## Next Steps
Probably need to find a way to convert the downloaded files (using tabula) to a pandas directory instead of writing to csv and then having pandas read it. 

Also need to add all the testing data into one single file for easy viewing. Then after will work on June and older data.

## Contents:
Testing numbers for each county daily. 

## Accuracy:
Most likely good but is estimated. 

Source: Pennsylvania National Electronic Disease Surveillance System (PA-NEDSS)

## Usefulness:
High? Idk... 
