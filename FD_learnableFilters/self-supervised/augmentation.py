from audiomentations import *

# In time-domain
# add多种数据增强方式
# 添加多种方式有助于提升模型效果，但不能影响音频的情感色彩。

# augmentation = SomeOf((2, 3),
augmentation = OneOf(
    [
        # """添加高斯噪音"""
        AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=40.0, p=0.5),
        # """添加高斯噪音"""
        # AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=40.0, p=1),

        # 将音频乘以一个随机的振幅因子来减少或增加音量。这种技术可以帮助一个模型在某种程度上对输入音频的整体增益不产生影响。
        Gain(min_gain_in_db=-12.0, max_gain_in_db=12.0, p=1),

        # 淡入淡出， 在随机时间内逐渐增大或减少音量。
        GainTransition(min_gain_in_db=-24.0,
                       max_gain_in_db=6.0,
                       min_duration=0.2,
                       max_duration=6.0,
                       duration_unit="seconds",
                       p=1),

        # 在不改变节奏的情况下将声音向上或向下移动
        PitchShift(min_semitones=-4.0, max_semitones=4, p=1),

        # 把音频样本倒过来，颠倒它们的极性。换句话说，将波形乘以-1，所以负值变成正值，反之亦然。当单独播放时，其结果与原始的声音是一样的。
        # 然而，在音频数据增强的背景下，这种转换在训练相位感知的机器学习模型时是有用的。
        PolarityInversion(p=1.0),

        # 将音频倒置。 类似于图像领域的数据增强
        Reverse(p=1),
        # 向前或向后移动样品，有或没有翻转。
        Shift(min_fraction=-0.5,
              max_fraction=0.5,
              rollover=True,
              fade=False,
              fade_duration=0.01,
              p=1),
        # 使音频中随机选择的部分无声。
        TimeMask(min_band_part=0.0,
                 max_band_part=0.5,
                 fade=False,
                 p=1.0),
        # 改变信号的速度或持续时间而不改变音高。
        TimeStretch(min_rate=0.8,
                    max_rate=1.25,
                    leave_length_unchanged=True,
                    p=1.0),
    ])
