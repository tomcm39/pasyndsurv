#mcandrew

import sys
import numpy as np
from epiweeks import Week, Year

class week(object):
    def __init__(self,modelweek=None,epiweek=None):
        from epiweeks import Week, Year
        self.refew        = Week(1970,1)
        self.refmodelweek = 0

        if modelweek is None and epiweek is None:
            raise self.NoData("Please enter either a model week or an epiweek")
        elif modelweek is not None and epiweek is not None:
            raise self.Toomuchdata("Please ejust one of a model week or an epiweek")
        elif modelweek is None:
            self.epiweek = epiweek
            if type(epiweek) is str:
                self.epiWeek = Week(int(epiweek[:4]),int(epiweek[4:]))
            elif type(epiweek) is int:
                epiweek = str(epiweek)
                self.epiWeek = Week(int(epiweek[:4]),int(epiweek[4:]))
            self.fromEpiWeek2ModelWeek()
        elif epiweek is None:
            self.modelweek = modelweek
            self.fromModelWeek2EpiWeek()
            self.epiWeek = Week(int(str(self.epiweek)[:4]),int(str(self.epiweek)[4:]))

        self.year = self.epiWeek.year
        self.week = self.epiWeek.week

        self.toFrom40Week()
            
    class NoData(Exception):
        pass
    class Toomuchdata(Exception):
        pass

    def fromModelWeek2EpiWeek(self):
        """ Convert model week to epiweek (CDC standard, MMWR Week)
        """
        eWeek = (self.refew+self.modelweek)
        eWeek = "{:04d}{:02d}".format(eWeek.year,eWeek.week)
        self.epiweek = int(eWeek)

    def fromEpiWeek2ModelWeek(self):
        """ Convert epiweek to model week, defined as the number of epidemic weeks from 197001
        """
        week = self.epiWeek
        numWeeks=0
        w = self.refew
        while True:
            if w.year < week.year:
                numWeeks+=Year(w.year).totalweeks
                w = Week(w.year+1,1)
            else:
                break
        while w < week:
            numWeeks+=1
            w+=1
        self.modelweek = numWeeks

    def toFrom40Week(self):
        """ A variable from40 is created as the number of weeks from the current year or, if epiweek's week is less than 40, the past year's epidemic week number 40 to epiweek.
        """
        yr,wk = self.year,self.week
        if wk > 40:
            self.from40 = wk-40
        else:
            _ref40week = Week(yr-1,40)
            _from40 = 0
            while _ref40week < self.epiWeek:
                _ref40week+=1
                _from40+=1
            self.from40 = _from40
