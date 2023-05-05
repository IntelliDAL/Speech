import os
import subprocess
import shutil


file = "/home/eric/mechanicalDisk/ward/normal_control"
audio_path_raw = "/home/eric/mechanicalDisk/ward/normal_control"

audio_path_wav = "/home/eric/mechanicalDisk/ward/normal_control_resample/"

audio_type = "wav"

if os.path.exists(audio_path_wav):
    shutil.rmtree(audio_path_wav)
os.mkdir(audio_path_wav)
Dir = os.listdir(file)

for each_file in Dir:
    print(each_file)
    audio_raw = os.path.join(audio_path_raw, each_file)
    audio_path = audio_path_wav + each_file.split(".")[0] + "." + audio_type
    print(each_file + "音频数据处理中...")
    subprocess.getoutput("ffmpeg -loglevel quiet -y -i %s -acodec pcm_s16le -ac 1 -ar 16000 %s" % (audio_raw, audio_path))