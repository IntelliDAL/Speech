import os
import argparse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import glob
import wave
import numpy as np
import pickle

# 疾病标签的映射
labeldict = {
    'hap': 0, 'ang': 1, 'neu': 2, 'sad': 3, 'exc': 4, 'fru': 5, 'fea': 6, 'sur': 7
}

def data_segmentwd(raw_data, window_size, strides_size):  # 数据分布
    windowList = []
    start = 0
    end = window_size
    while True:
        x = raw_data[start:end]
        if len(x) < window_size:
            break
        windowList.append(x)
        if end >= raw_data.shape[0]:
            break
        start += strides_size
        end += strides_size
    window = np.array(windowList)
    return np.array(window)


def read_wavefile(filename):
    # 开始读取wav文件
    file = wave.open(filename, 'r')
    params = file.getparams()  # 获取得到的所有参数
    n_channels, samp_with, fram_rate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / fram_rate)
    file.close()
    return wave_data, time, fram_rate


def read_IEMocap(args):
    # 设置数据库的路径
    data_DIR = "/home/idal-01/data/IEMCOAP"
    # 设置训练数据和标签
    train_data = []
    train_label = []
    times = []
    audio_length = args.audio_length
    stride = 0.1 * audio_length
    # 开始对数据库进行遍历过程
    for speaker in os.listdir(data_DIR):
        print(speaker)
        """
        直接开始对每个进行搜索
        并且设置缩小的范围 需要的数据进行查找
        """
        # 因为还有其他的不是所需要的文件所以直接进行排除
        if speaker[0] == 'S':
            # 语音存储的文件夹
            speech_subdir = os.path.join(data_DIR, speaker, "sentences/wav")
            # 语音标记的文件夹
            speech_labledir = os.path.join(data_DIR, speaker, "dialog/EmoEvaluation")
            # 将训练文件夹也保存
            speech_file_dir = []
            for sess in os.listdir(speech_subdir):
                # sess 代表的是每个单独的文件夹 里面包含着每个单独的txt文件所以需要单独读取
                lable_text = speech_labledir + "/" + sess + ".txt"
                # 获取到了 然后开始读取，这时要知道 读取文件需要用个list 或者是字典来进行存取
                emotion_lable = {}

                with open(lable_text, 'r') as txt_read:
                    """
                    这里表达的是，文件读取第一行 看第一行如果有文件则进行保存对应的标签和结果 其中包含标注信息
                    直到文件最后读取结束
                    """
                    while True:
                        line = txt_read.readline()
                        if not line:
                            break
                        if (line[0] == '['):
                            t = line.split()
                            emotion_lable[t[3]] = t[4]
                # --------------------------------------------------------
                """
                读取所有的音频文件
                """
                wava_file = os.path.join(speech_subdir, sess, '*wav')
                files = glob.glob(wava_file)  # glob 主要是将目标的所有 来进行返回一个list集合
                Scaler = StandardScaler()
                for filename in files:
                    # 开始读取speech文件内的信息了 文件标签 存储数据内容
                    wavaname = filename.split("/")[-1][:-4]  # 得到文件名
                    emotion = emotion_lable[wavaname]  # 通过对应来得到情感的对应标记
                    # 这里开始筛选是不是你需要的文件类型 比如你只想要hap ang neu sad 不要fear 那就可以不用把这个fear放入
                    # if emotion in ['hap', 'ang', 'neu', 'sad']:
                    if emotion in ['hap', 'ang', 'neu', 'sad', 'exc', 'fru', 'fea', 'sur']:
                        # data standard
                        data, time, rate = read_wavefile(filename)
                        data = Scaler.fit_transform(data.reshape(-1, 1))
                        data = data.squeeze().astype(np.float32)

                        time_len = len(time) / rate
                        # 开始对不满足时间少于300的进行padding 0
                        times.append(time_len)
                        if time_len >= audio_length:
                            print("开始对{}文件的计算".format(filename))
                            print(time_len)
                            padding_data = data_segmentwd(data, int(audio_length * rate), int(stride * rate))
                            m, _ = padding_data.shape
                            train_data.append(padding_data)
                            train_label.extend(np.array(labeldict.get(emotion)).repeat(m))
                            speech_file_dir.append(filename)
                        # else:
                        #     padding_data = data
                        #     # 后面补充0
                        #     padding_data = np.pad(padding_data, (0, (audio_length * rate - padding_data.shape[0])), 'constant', constant_values=0)
                        #     train_data.extend([padding_data])
                        #     train_label.extend([labeldict.get(emotion)])
                        #     speech_file_dir.append(filename)
    numbers = len(train_label)
    num_hap = train_label.count(0)
    num_ang = train_label.count(1)
    num_neu = train_label.count(2)
    num_sad = train_label.count(3)
    num_exc = train_label.count(4)
    num_fru = train_label.count(5)
    num_fea = train_label.count(6)
    num_sur = train_label.count(7)
    # 写入pkl文件中
    print("开始写入文件中")
    f = open('./IEMOCAP.pkl', 'wb')
    # 将数据 放入pickle中保存
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    pickle.dump((train_data, train_label, times, speech_file_dir), f)
    f.close()
    print("写入完毕")
    pass


def read_picklefile():
    f = open('./IEMOCAP.pkl', 'rb')
    train_data, train_lable, times, speech_file = pickle.load(f)
    print(len(train_lable))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', type=int, default=16000)
    parser.add_argument('--audio_length', type=int, default=6.46)
    args = parser.parse_args()
    read_IEMocap(args)
    # read_picklefile()
    print("6-kinds")
