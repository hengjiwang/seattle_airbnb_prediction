import datetime, calendar
import pandas as pd
from dateutil.parser import parse

class DatePreprocessing:
    def __init__(self, path):
        self.path = path
        self.holidays = ['NewYear','MartinLK','President','Memorial','Independence','Labor','Columbus','Veterans','Thanksgiving','Christmas']
        self.week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        self.month =  ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        self.year = [2018,2019]


    def holiday(self, year):
        #fixed holidays
        fixmd = [[1,1],[7,4],[11,11],[12,25]]
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
        df = self.clean_calendar()
        return df
    
    def clean_calendar(self):
        df = pd.read_csv(self.path).dropna().drop(columns='available').reset_index(drop=True)
        df = df[['listing_id','date','price']] #only keep these three columns
        df['price'] =  df['price'].apply(lambda x: x.replace('$','').replace(',','')).astype('float')   
        df['date'] = pd.to_datetime(df['date'])
        df = self.add_scraped(df)
        return df

    def add_weeks(self, df):
        #day of week
        week = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        df['week'] = df.date.dt.dayofweek
        for i,w in enumerate(self.week):
            df[w]=0
            df.loc[df['week']==i,w]=1
        #df['FriSat'] = df.apply(lambda x: int(x['week']==4 or x['week']==5),axis=1)
        return df

    def add_months(self, df):
        # month = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        for i,m in enumerate(self.month):
            df[m]=0
            df.loc[df.date.dt.month==i+1,m]=1
        return df

    def add_weather(self,df):
        weather = pd.read_excel('../seattle_weather.xlsx').values
        df['aveT'] = 0
        df['precipitation'] = 0
        for i,m in enumerate(self.month):
            df.loc[df[m]==1,['aveT','precipitation']]=weather[i]
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
    
    def add_scraped(self,df):
        scraped = parse([x for x in self.path.split('/') if x.startswith('20')][0].replace('_',''))
        df['days_from_scraped'] = (df['date']-scraped).dt.days
        #df['days_from_scraped'] = ListingsPreprocessing('na').normalize(df['days_from_scraped']+1)
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
    
    
    def drop_dupweek(self,df):
        myholi = self.holidays
        myholi.append('ChristmasHolidays')
        df_holi = df[(df[myholi]==1).any(axis=1)].copy()
        df_nholi = df[(df[myholi]==0).all(axis=1)].copy()    
        
        df_nholi['ym'] = df_nholi['date'].dt.strftime('%y%m')
        mark = ['listing_id','week','ym']
        df_nholi.drop_duplicates(subset=mark,keep='first',inplace=True)  
        df_nholi.drop(columns='ym',inplace=True)
        df = pd.concat([df_holi,df_nholi],ignore_index=True,sort=False)
        return df
    
    def extract_feature(self, df):
        
        df = self.add_weeks(df)
        #df = self.add_christmas_holidays(df)
        df = self.add_other_holidays(df)
        #df = self.get_average(df)
        #df = self.drop_dupweek(df)   
        df = self.add_months(df)
        df = self.add_years(df)
        df = self.add_weather(df)
        df = df.drop(columns=['week','date'])
        return df


    def do_preprocessing(self):
        df = self.load()
        df = self.extract_feature(df)
        #df = df.drop(columns=['week','date'])
        #df = df.drop(columns=self.week)
        return df




