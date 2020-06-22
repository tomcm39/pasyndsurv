#mcandrew;

#code options-------------------------
PYTHON = \python3
PYTHON_OPTS = -W ignore
#-------------------------------------

# Update Number of new cases data
# --------------------------------------------------------------------------------------
updateData: updateCovidCastCases updateJHUCSSE updateLehighPADOH updateCOVIDtracking \
            mergeCasesData buildTrainingData

updateCovidCastCases:
	cd ./data/cases/covidcast/ && $(PYTHON) $(PYTHON_OPTS) downloadData.py && \
	echo "CoviCastData updated" && cd ../../../

updateJHUCSSE:
	cd ./data/cases/jhuCSSE/ && $(PYTHON) $(PYTHON_OPTS) downloadData.py && \
	echo "JHU CSSE updated" && cd ../../../

updateLehighPADOH:
	cd ./data/cases/LehighDOHdata/ && $(PYTHON) $(PYTHON_OPTS) downloadData.py && \
	echo "Lehigh PA DOH updated" && cd ../../../

updateCOVIDtracking:
	cd ./data/cases/covidtracking/ && $(PYTHON) $(PYTHON_OPTS) downloadData.py && \
	echo "COVIDTracking updated" && cd ../../../

mergeCasesData:
	cd ./data/cases && $(PYTHON) $(PYTHON_OPTS) mergeCaseData.py && \
	echo "merged cases data together" && cd ../../

buildTrainingData:
	cd ./data/cases && $(PYTHON) $(PYTHON_OPTS) buildTrainingData.py && \
	echo "training data built" && cd ../../

# train models
# --------------------------------------------------------------------------------------
scoreAndForecast:
	cd ./scores/ && $(PYTHON) $(PYTHON_OPTS) scoreEachModel.py && \
	echo "Models scored and forecasts built" && cd ../ 

# train ensemble
# --------------------------------------------------------------------------------------
