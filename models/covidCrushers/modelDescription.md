Team name: covidCrushers
Team members: Kenny Lin, Alex Kline

Model Description Short:

The goal is to create simulated forecast data for COVID-19 cases using a Holt-Winter model and the various forecast models currently being developed by the team.  This data will be used as training data for the neural network. As raw data does not exist and traditional forecasting methods have been shown to be superior to machine learning models, perhaps by aggregating the results of the more time proven classical models, the neural network might be trained to make more accurate predictions than otherwise.

Model Description Long:

 - Model assumptions

This model proceeds under the assumption that simulated data is an adequate substitue for real data.

The logical thought would be that forecasts produced by the neural network would then only depend on the accuracy of the forecasted models themselves. However, this might not be the case as statistical models often predict better accuracy than machine learning models. By aggregating multiple statistical models, it might therefore be possible to produce a better model than the parts. 

However, while ensemble models have shown to be more accurate than the models they are trained, this isn't always neccessarily the case as with case 23 and 24 here: http://thomasmcandrew.com/classes/2019_stat340_MHC/public/

This model also assumes that COVID/Influenza data from other countries and states are different enough from Pennsylvania to disallow their qualification as training data.

Neural networks have multiple hidden layers and use gradient descent to make decisions based on given data. To work well however, they generally need a substantial amount of training data to “learn” from. COVID-19, unfortunately, is an entirely new disease and was only recently discovered back in December/January. Consequently, there is an inherent lack of data in the first place. 

Further worsening matters, every country’s government has handled the outbreak entirely differently to varying degrees of success. Many Asian countries, for example, have employed contract tracing and lockdowns successively and have managed to mostly stamp out the disease. Their prevalent culture of wearing masks whenever feeling unwell has also helped reduce transmission levels. 

In Europe, the presence of universal health care may have also affected transmission reporting and death rates, perhaps leading to more testing and more availability of care to the lower class. In Italy, however, the disproportionate presence of the elderly population, resulted in tragically high death rates. 

Because so many factors are variable, COVID data from other countries should not be used to train the neural network for forecasting the state of Pennsylvania. Even among states in the US, the disjointed response to the outbreak as well as the difference in severity levels between the states would invalidate the data for training. 

A potential solution would be to use past influenza data to train the network. This is possible as influenza has a long history of data collection and spread within Pennsylvania. However, there are several important distinctions between influenza and COVID-19. First, annual vaccines are developed for influenza and a large proportion of the US is inoculated to reduce the impact of seasonal epidemics. In addition, the influenza is largely seasonal with common cases mostly occurring around fall/winter. As it is summer and the spread of the novel coronavirus has not been significantly impacted, transmission relevance to influenza may be limited. 


 - Model pros and cons



 - What data sources did you use


 - What variables from those data sources did you use. Why?

