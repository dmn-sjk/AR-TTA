from shift_dev.types import WeathersCoarse, TimesOfDayCoarse


WEATHERS_SEQUENCE = [WeathersCoarse.clear,
                     WeathersCoarse.cloudy,
                     WeathersCoarse.overcast,
                     WeathersCoarse.rainy,
                     WeathersCoarse.foggy]

TIMEOFDAY_SEQUENCE = [TimesOfDayCoarse.daytime, 
                      TimesOfDayCoarse.dawn_dusk,
                      TimesOfDayCoarse.night]
