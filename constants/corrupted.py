SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS = ["shot_noise", "motion_blur", "snow", "pixelate", "gaussian_noise", "defocus_blur", "brightness", "fog", \
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast", "jpeg_compression", "elastic_transform"]

STANDARD_DOMAINS_SEQ = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression"
]

REPETITIVE_DOMAINS_SEQ = [
    *STANDARD_DOMAINS_SEQ, # one loop
    *STANDARD_DOMAINS_SEQ, # 2nd loop
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "gaussian_noise",
    "shot_noise",
    "impulse_noise"
]

LONG_DOMAINS_SEQ = [
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "gaussian_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise",
    "impulse_noise"
]