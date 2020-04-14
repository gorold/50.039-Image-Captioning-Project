from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, RandomSizedCrop, ElasticTransform, RGBShift, GaussianBlur, GlassBlur, ToSepia, Cutout, RandomScale, Resize
)

def get_augmentations(img_size):
    return Compose([
        Resize(height = int(img_size * 1.5), width = int(img_size * 1.5), p = 1),
        RandomSizedCrop(min_max_height = (int(img_size * 0.9), img_size), height = img_size, width = img_size, always_apply= True, p = 1),
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.4),
        OneOf([
            GlassBlur(p= 1),
            GaussianBlur(p = 1),
            MotionBlur(p=1),
            MedianBlur(blur_limit=3, p=1),
            Blur(blur_limit=3, p=1),
        ], p=0.4),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p = 1),
            ElasticTransform(),
            GridDistortion(p = 1),
            IAAPiecewiseAffine(p = 1),
        ], p = 0.4),
        OneOf([
            CLAHE(clip_limit=2), # Histogram Equalization
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
            RGBShift()
        ], p = 0.4),
        HueSaturationValue(p=0.3),
        ToSepia(p = 0.2),
        Cutout(p = 0.2),
        RandomScale(p = 0.2)
    ])