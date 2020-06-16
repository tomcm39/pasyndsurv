#Model: Covid Cast

##API Link:
https://cmu-delphi.github.io/delphi-epidata/api/covidcast.html

##Database Description:
COVIDcast displays signals related to COVID-19 activity levels across the United States, derived from a variety of anonymized, aggregated data sources made available by multiple partners.

Each signal may reflect the prevalence of COVID-19 infection, mild symptoms, or more severe disease over time. Each signal can be presented at multiple geographic resolutions: state, county, and/or metropolitan area.

All these signals taken together may suggest heightened or rising COVID-19 activity in specific locations. They will provide useful inputs to CMU’s pandemic forecasting system (to be announced at a later date).

##Database Link:
https://covidcast.cmu.edu/?sensor=doctor-visits-smoothed_adj_cli&level=county&region=42003&date=20200611&signalType=value

##Contents:
Contains activae indicators of COVID-19 such as estimations of the percentage of doctors visits related to COVID like symptons, results from surveys of user symptons and symptons in community conducted via Facebook, and search trends on Google.

Combined map doesn't show actual number of cases but rather an estimated intensity scale for each region.

##Accuracy:
Not sure. The surveys, especially the ones that rely on word of mouth (i.e. knowing of someone with COVID like symptons) could be really inaccurate. I'm also not sure what they're basing their doctor visitation estimates for flulike illnesses on.  

##Usefulness:
Limited. Seems super unique and really interesting but not sure how/if we will factor it into our forecasts.

I guess it depends on what we're trying to forecast. For instance, if we're trying to understand the real picture of how many cases are in an area, this could be the only source that can estimate that infromation.
