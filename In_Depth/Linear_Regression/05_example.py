import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

def hours_of_daylight(date, axis=23.44, latitude=47.61):
    """Compute the hours of daylight for the given date"""
    days = (date - pd.datetime(2000, 12, 21)).days
    m = (1. - np.tan(np.radians(latitude))
         * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
    return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.

counts = pd.read_csv('Fremont.csv', index_col='Date', parse_dates=True)
weather = pd.read_csv('weather.csv'index_col='DATE', parse_dates=True)
daily = counts.resample('d').sum()
daily['totals'] = daily.sum(axis=1)
daily = daily[['Total']]
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i in range(7):
    daily[days[i]] = (daily.index.dayofweek == i).astype(float)

calendar = USFederalHolidayCalendar()
holidays = calendar.holidaysstart='2012', end='2016')
daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
daily['holiday'].fillna(0, inplace=True)
daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
daily[['daylight_hrs']].plot()
plt.ylim(8, 17)
plt.show()
