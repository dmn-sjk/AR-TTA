from datasets.shift_dev.types import TimesOfDayCoarse, WeathersCoarse


def get_domain_sequence(cfg):
    if cfg['dataset'] in ['cifar10c', 'imagenetc']:
        return [CorruptionDomain(corr, 5) for corr in CORRUPTIONS_SEQ]
    elif cfg['dataset'] == 'shift':
        domain_seq = []
        for timeofday in TIMEOFDAY_SEQ:
            for weather in WEATHERS_SEQ:
                # ommit source domain
                if timeofday == TimesOfDayCoarse.daytime and weather == WeathersCoarse.clear:
                    continue
                domain_seq.append(ShiftDomain(timeofday, weather))
        return domain_seq
    elif cfg['dataset'] == 'clad':
        return [CladDomain(i) for i in range(1, 6)]
    elif cfg['dataset'] == 'cifar10_1':
        return [DummyDomain()]
    else:
        raise ValueError(cfg['dataset'])


class Corruptions:
    gaussian_noise = "gaussian_noise"
    shot_noise = "shot_noise"
    impulse_noise = "impulse_noise"
    defocus_blur = "defocus_blur"
    glass_blur = "glass_blur"
    motion_blur = "motion_blur"
    zoom_blur = "zoom_blur"
    snow = "snow"
    frost = "frost"
    fog = "fog"
    brightness = "brightness"
    contrast = "contrast"
    elastic_transform = "elastic_transform"
    pixelate = "pixelate"
    jpeg_compression = "jpeg_compression"

SEVERITIES = [1, 2, 3, 4, 5]

CORRUPTIONS_SEQ = [
    Corruptions.gaussian_noise,
    Corruptions.shot_noise,
    Corruptions.impulse_noise,
    Corruptions.defocus_blur,
    Corruptions.glass_blur,
    Corruptions.motion_blur,
    Corruptions.zoom_blur,
    Corruptions.snow,
    Corruptions.frost,
    Corruptions.fog,
    Corruptions.brightness,
    Corruptions.contrast,
    Corruptions.elastic_transform,
    Corruptions.pixelate,
    Corruptions.jpeg_compression
]

WEATHERS_SEQ = [WeathersCoarse.clear,
                WeathersCoarse.cloudy,
                WeathersCoarse.overcast,
                WeathersCoarse.rainy,
                WeathersCoarse.foggy]

TIMEOFDAY_SEQ = [TimesOfDayCoarse.daytime, 
                TimesOfDayCoarse.dawn_dusk,
                TimesOfDayCoarse.night]

class Domain:
    def get_domain_dict(self):
        raise NotImplementedError
    
    def __str__(self):
        raise NotImplementedError

class ShiftDomain(Domain):
    def __init__(self, timeofday: TimesOfDayCoarse, weather: WeathersCoarse):
        self.timeofday = timeofday
        self.weather = weather
        
    def __str__(self):
        return f"{self.timeofday}_{self.weather}"
    
    def get_domain_dict(self):
        return {'weathers_coarse': [self.weather], 'timeofdays_coarse': [self.timeofday]}

class CorruptionDomain(Domain):
    def __init__(self, corruption: Corruptions, severity: int):
        self.corruption = corruption
        self.severity = severity
        
    def __str__(self):
        return f"{self.corruption}_{self.severity}"

    def get_domain_dict(self):
        return {'corruption': self.corruption, 'severity': self.severity}
    
class CladDomain(Domain):
    def __init__(self, num: int):
        self.num = num
        
    def __str__(self):
        return f"T{self.num}"

    def get_domain_dict(self):
        return {'task_id': self.num}

class DummyDomain(Domain):
    def __str__(self):
        return ""

    def get_domain_dict(self):
        return {}
 