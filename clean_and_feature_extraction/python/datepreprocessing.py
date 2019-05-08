import datetime,calendar
import pandas as pd
from calendar_clean_exploration import clean_calendar

class DatePreprocessing:
    def __init__(self, path):
        self.path = path
        self.holidays = ['NewYear','MartinLK','President','Memorial','Independence','Labor','Columbus','Veterans','Thanksgiving']
        self.week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        self.month =  ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        self.year = [2018,2019]


    def holiday(self, year):
        #fixed holidays
        fixmd = [[1,1],[7,4],[11,11]]
        holidays = [datetime.datetime(year,x[0],x[1]) for x in fixmd]
        #floating holidays
        #'2,3,1': the third Monday of Feburary;'5,-1,1': the last Monday of May 
        floatmwd = ['1,3,1','2,3,1','5,-1,1','9,1,1','10,2,1','11,4,4']
        holidays.extend([self.flholiday(year,x) for x in floatmwd])
        return sorted(holidays)
    
    def flholiday(self, y,mwd):
        #input year and month-week-weekday
        m,w,d = list(map(int,mwd.split(',')))
        d = d-1
        wday, mrange = calendar.monthrange(y, m) #firstDayWeekDay,monthRange
        if w>0:
            if wday<=d:
                day = 1+7*(w-1)+d-wday
            else:
                day = 1+7*w-(wday-d)
        else:
            wday = datetime.datetime(y,m,mrange).weekday()
            if d<=wday:
                day = mrange+7*(w+1)-(wday-d)
            else:
                day = mrange+7*w+(d-wday)
        return datetime.datetime(y,m,day)

    def load(self):
        df = clean_calendar(self.path)
        df['week'] = df.date.dt.dayofweek
        return df

    def add_weeks(self, df):
        #day of week
        # week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        for i,w in enumerate(self.week):
            df[w]=0
            df.loc[df['week']==i,w]=1
        return df

    def add_months(self, df):
        # month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        for i,m in enumerate(self.month):
            df[m]=0
            df.loc[df.date.dt.month==i+1,m]=1
        return df

    def add_years(self, df):
        # year = [2018,2019]
        for y in self.year:
            df[y]=0
            df.loc[df.date.dt.year==y,y]=1
        return df
    
    def add_christmas_holidays(self, df):
        df['ChristmasHolidays']=0
        df.loc[(df.date.dt.month==12) & (df.date.dt.day>=25),'ChristmasHolidays']=1
        return df

    def add_other_holidays(self, df):
        # holidays = ['NewYear','MartinLK','President','Memorial','Independence','Labor','Columbus','Veterans','Thanksgiving']
        holiday18 = self.holiday(2018)
        holiday19 = self.holiday(2019)
        for i,h in enumerate(self.holidays):
            df[h]=0
            df.loc[df.date==holiday18[i],h]=1
            df.loc[df.date==holiday19[i],h]=1
        return df

    def get_average(self, df):
        myholi = ['ChristmasHolidays']
        myholi.extend(self.holidays)
        df_nholi = df[(df[myholi]==0).all(axis=1)].drop(columns=myholi)

        gmark = ['listing_id']
        gmark.extend(self.year)
        gmark.extend(self.month)
        gmark.extend(self.week)

        result = df_nholi.groupby(gmark)['price'].mean().to_frame()
        result.reset_index(inplace=True)
        df_holi = df[(df[myholi]==1).any(axis=1)]
        df = pd.concat([df_holi,result],ignore_index=True,sort=False).fillna(0)
        return df


    def do_preprocessing(self):
        df = self.load()
        df = self.add_weeks(df)
        df = self.add_months(df)
        df = self.add_years(df)
        df = self.add_christmas_holidays(df)
        df = self.add_other_holidays(df)
        df.drop(columns=['week','date'],inplace=True)
        df = self.get_average(df)
        return df


