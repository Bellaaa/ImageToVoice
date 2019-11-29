import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

def load_voice(voice_item):
    voice_data = np.load(voice_item['filepath'])
    voice_data = voice_data.T.astype('float32')
    voice_label = voice_item['label_id']
    return voice_data, voice_label

def load_face(face_item):
    face_data = Image.open(face_item['filepath']).convert('RGB').resize([64, 64])
    face_data = np.transpose(np.array(face_data), (2, 0, 1))
    face_data = ((face_data - 127.5) / 127.5).astype('float32')
    face_label = face_item['label_id']
    return face_data, face_label

def getRandomizedVoice(voice_item, crop_nframe):
    voice_data, voice_label = load_voice(voice_item)
    assert crop_nframe <= voice_data.shape[1]
    pt = np.random.randint(voice_data.shape[1] - crop_nframe + 1)
    voice_data = voice_data[:, pt:pt + crop_nframe]
    return voice_data, voice_label

def getRandomizedFace(face_item):
    face_data, face_label = load_face(face_item)
    if np.random.random() > 0.5:
        face_data = np.flip(face_data, axis=2).copy()
    return face_data, face_label

def reload_batch_voice(voice_items, crop_nframe):
    tmp_list = [torch.from_numpy(getRandomizedVoice(item, crop_nframe)[0]).unsqueeze(0) for item in voice_items]
    return torch.cat(tmp_list, dim=0)

def reload_batch_face(voice_items):
    tmp_list = [torch.from_numpy(getRandomizedFace(item)[0]).unsqueeze(0) for item in voice_items]
    return torch.cat(tmp_list, dim=0)


class VoiceDataset(Dataset):
    def __init__(self, voice_list, nframe_range):
        self.voice_list = voice_list
        self.crop_nframe = nframe_range[1]

    def __getitem__(self, index):
        voice_item = self.voice_list[index]
        return getRandomizedVoice(voice_item, self.crop_nframe)

    def __len__(self):
        return len(self.voice_list)

class FaceDataset(Dataset):
    def __init__(self, face_list):
        self.face_list = face_list

    def __getitem__(self, index):
        face_item = self.face_list[index]
        return getRandomizedFace(face_item)

    def __len__(self):
        return len(self.face_list)
