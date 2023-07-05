from shift_dev.types import WeathersCoarse, TimesOfDayCoarse


WEATHERS_SEQUENCE = [WeathersCoarse.clear,
                     WeathersCoarse.cloudy,
                     WeathersCoarse.overcast,
                     WeathersCoarse.rainy,
                     WeathersCoarse.foggy]

TIMEOFDAY_SEQUENCE = [TimesOfDayCoarse.daytime, 
                      TimesOfDayCoarse.dawn_dusk,
                      TimesOfDayCoarse.night]

STANDARD_SHIFT_MIX_DOMAIN_SEQUENCE = []
for timeofday in TIMEOFDAY_SEQUENCE:
    for weather in WEATHERS_SEQUENCE:
        if timeofday == TimesOfDayCoarse.daytime and weather == WeathersCoarse.clear:
            continue
        STANDARD_SHIFT_MIX_DOMAIN_SEQUENCE.append(f"{timeofday}_{weather}")