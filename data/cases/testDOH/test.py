# klin

import sys
sys.path.append('../..')

import tabula
import pandas as pd
from bs4 import BeautifulSoup as soup
from urllib.request import urlopen  

# modules from prof mcandrew
from downloadHelper.downloadtools import timestamp
from downloadHelper.downloadtools import listPACounties

# METHODS #

# calculate total tests from the DOH data
def cal_tot_tests(data_frame):
    test_col = data_frame['Confirmed'] + \
               data_frame['PersonsWithNegativePCR']
    return test_col

def add_test_col(data_frame, test_col):
    # estimated total tests
    data_frame['Est_Tot_Tests'] = test_col  

# inserts fips row to the county 
def add_fips(df, countyList):
    df.insert(1, 'Fips', countyList)

# inserts repeated date row to a df
def add_date(df, date):
    df.insert(0, 'Date', date)

# extract date + url from html list of links
# input: html string, link list, date list 
def extract_data(raw_links, list_dates, dict_links):
    for link in raw_links:
        str_link = str(link)

        # indexes
        date_start = str_link.find('_') + 1
        date_end = str_link.find('.') 

        # obtain date of entry from url 
        date = str_link[date_start: date_end]
        full_link = "https://www.health.pa.gov" + str(link['href'])

        # update dates and links
        list_dates.append(date)
        dict_links.update({date: full_link})
    #change to ascending order (in time)
    list_dates.reverse()


# converts pdfs into df and adds to dict_df 
def downloadData(list_dates, dict_df):
    for curr_date in list_dates:
        return_file = 'data/dohData_' + curr_date + '.csv'
        df_pdf = tabula.read_pdf(dict_links[curr_date], pages="all")
        print(type(df_pdf))
        print(df_pdf)
        # date is the key, df is value
        dict_df.update({curr_date: df_pdf})
        #tabula.convert_into(dict_links[curr_date], return_file, pages="all")
        print("Read file: " + return_file)

# checks two input strings and checks if they are equal
# for exception files
def verifyHeader(current_header, standard_header, curr_file):
    if ( (current_header != standard_header)):
        print()
        print("Skipped augmentation of " + curr_file + \
              " due to nonstandard header.")
        print("File Header: ")
        print(current_header)
        print()
        return 1 # true (is not a valid header)
    return 0 # false (is a valid header)

# aggregate datas files for each date
def aggregateFiles(list_dates):
    prev_iter = 0
    curr_iter = 0
    
    standard_header = ['County', 'Region', 'Cases', 'Confirmed', \
                            'Probable', 'PersonsWithNegativePCR']

    for curr_date in list_dates:

        prev_date = list_dates[prev_iter]
        # these are file names
        curr_file = 'data/dohData_' + curr_date + '.csv'
        #prev_file = 'data/dohData_' + prev_date + '.csv'

        curr_df = dict_df.get(curr_date)
        prev_df = dict_df.get(prev_date)

        # skip non standard files
        print(curr_df.columns)
        current_header = list(curr_df.columns)
        if (verifyHeader(current_header, standard_header, curr_file)):
            curr_iter += 1
            continue
        else:
            prev_iter = curr_iter
            curr_iter += 1
        
        # augments data from the DOH with test totals and fip numbers
        curr_tests = cal_tot_tests(curr_df)
        prev_tests = cal_tot_tests(prev_df)

        add_test_col(curr_df, curr_tests)
        add_fips(curr_df, county2fips)

        # change in num of tests from last date
        change_tests = curr_tests - prev_tests

        curr_df.insert(len(curr_df.columns), 'Date', curr_date)
        curr_df.insert(len(curr_df.columns), 'Last Date', prev_date)
        curr_df.insert(len(curr_df.columns), 'Change From Last Date', \
                        change_tests)

        curr_df.to_csv(curr_file, index=False)
        curr_df.to_csv(curr_file, index=False)

        print("Augmented file: " + curr_file)

# reads in all files 
'''
def aggregateTests(list_dates):
    for curr_date in list_dates:
'''
# END METHODS #

#July Archive
july_url = '''https://www.health.pa.gov/topics/disease/coronavirus/Pages/Archives.aspx'''

uClient = urlopen(july_url)
page_html = uClient.read()
uClient.close()
page_soup = soup(page_html, 'html.parser')

# all links to case data
raw_links = page_soup.find_all("a", title = "County case counts by date")

# organizies all fips with their county names
# stolen from prof mcandrew
fips2county = listPACounties()
county2fips = [fip for (fip, county) in fips2county.items()]

# store date + url from html to list_dates, dict_links
list_dates = []
dict_links = {}
dict_df = {}
extract_data(raw_links, list_dates, dict_links)

downloadData(list_dates, dict_df)
print()
aggregateFiles(list_dates)