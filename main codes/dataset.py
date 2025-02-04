from torch.utils.data.dataset import Dataset
import torchaudio
import librosa
import os
import torch

class PVQDDataset(Dataset):
    def __init__(self, wavdir, label_list, feature_extractor, multi_indicator=True):
        self.label_lookup = {}  # Initialize the dictionary here
        f = open(label_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            # print(parts)
            if multi_indicator == False:
                label = float(parts[1])
            else:
                label = [float(i) for i in parts[1:]]
            self.label_lookup[wavname] = label

        self.wavdir = wavdir
        self.wavnames = sorted(self.label_lookup.keys())
        self.feature_extractor = feature_extractor


    #__getitem__() method: a special (magic) method that allows instances of a class to support indexing and item retrieval like lists, tuples, or dictionaries.
    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.label_lookup[wavname]
        return wav, score, wavname
    
    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):  ## zero padding
            # Groups the first elements together, the second elements together, and so on.
            wavs, scores, wavnames = zip(*batch)
            wavs = list(wavs)
            # The key argument specifies a function (lambda x: x.shape[1]) that is applied to each element of the list wavs.
            max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
            output_wavs = []
            mel_specgrams = []
            asr_mel_features = []
            transform = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,n_fft =400,win_length = 400,hop_length =320
            ,n_mels=40, center=False)
            for wav in wavs:
                amount_to_pad = max_len - wav.shape[1]
                padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
                mel_specgram = transform(padded_wav).squeeze(0)
                delta = torchaudio.functional.compute_deltas(mel_specgram)
                delta2 = torchaudio.functional.compute_deltas(delta)
                mel_specgram = torch.cat((mel_specgram,delta,delta2),dim=0)
              
                output_wavs.append(padded_wav)
                mel_specgrams.append(mel_specgram)
                padded_wav = padded_wav.squeeze(0)

                # Detaches the tensor from the computation graph.
                # This is essential to prevent gradients from being tracked during backward propagation.
                wav = wav.to('cpu').detach().numpy().copy()
                asr_mel_feature = self.feature_extractor(wav, return_tensors="pt",sampling_rate=16000).input_features
                asr_mel_features.append(asr_mel_feature)

            
            output_wavs = torch.stack(output_wavs, dim=0)
            mel_specgrams = torch.stack(mel_specgrams,dim=0)
            scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
            asr_mel_features = torch.stack(asr_mel_features,dim=0)
            return asr_mel_features, output_wavs,mel_specgrams, scores, wavnames
      
class DBSDataset(Dataset):
    def __init__(self, dbs_list,feature_extractor, multi_indicator):
        self.grbas_lookup = { }
        f = open(dbs_list, 'r')
        for line in f:
            parts = line.strip().split(',')
            wavname_a = parts[0]
            if multi_indicator:
                grbas = [float(parts[5]),float(parts[6]),float(parts[7]),float(parts[8]),float(parts[9])]
            else:
                grbas =  [float(parts[5])]
            self.grbas_lookup[wavname_a] = grbas

        self.wavnames = sorted(self.grbas_lookup.keys())
        self.feature_extractor = feature_extractor


    def __getitem__(self, idx):
        wavname = self.wavnames[idx]
        wavpath_a = wavname
        base_dir,wave_dir = wavname.rsplit("/",2)[0],wavname.rsplit("/",2)[2]
        # print(base_dir,wave_dir)
        wavpath_i = os.path.join(base_dir,"i",wave_dir.rsplit("_",1)[0]+".wav")
        wavpath_u = os.path.join(base_dir,"u",wave_dir.rsplit("_",1)[0]+".wav")
        wavpath_e = os.path.join(base_dir,"e",wave_dir.rsplit("_",1)[0]+".wav")
        wavpath_o = os.path.join(base_dir,"o",wave_dir.rsplit("_",1)[0]+".wav")
        wav_a = torchaudio.load(wavpath_a)[0]
        _,T = wav_a.shape
        wav_a = wav_a[:,T//4:T//4+T//2]

        wav_i = torchaudio.load(wavpath_i)[0]
        _,T = wav_i.shape
        wav_i = wav_i[:,T//4:T//4+T//2]

        wav_u = torchaudio.load(wavpath_u)[0]
        _,T = wav_u.shape
        wav_u = wav_u[:,T//4:T//4+T//2]

        wav_e = torchaudio.load(wavpath_e)[0]
        _,T = wav_e.shape
        wav_e = wav_e[:,T//4:T//4+T//2]

        wav_o = torchaudio.load(wavpath_o)[0]
        _,T = wav_o.shape
        wav_o = wav_o[:,T//4:T//4+T//2]
        score = self.grbas_lookup[wavname]
        return wav_a,wav_i,wav_u,wav_e,wav_o, score, wavname

    def __len__(self):
        return len(self.wavnames)

    def collate_fn(self, batch):  ## zero padding
        wav_a,wav_i,wav_u,wav_e,wav_o, score, wavname = zip(*batch)

        output_wavs = torch.cat((wav_a[0],wav_i[0],wav_u[0],wav_e[0],wav_o[0]),dim=-1)

        transform = torchaudio.transforms.MelSpectrogram(sample_rate = 16000,n_fft =400,win_length = 400,hop_length =320
            ,n_mels=40, center=False)
        mel_specgram = transform(output_wavs).squeeze(0)
        delta = torchaudio.functional.compute_deltas(mel_specgram)
        delta2 = torchaudio.functional.compute_deltas(delta)
        mel_specgram = torch.cat((mel_specgram,delta,delta2),dim=0)
        output_wav = output_wavs.to('cpu').detach().numpy().copy()

        asr_mel_feature = self.feature_extractor(output_wav, return_tensors="pt",sampling_rate=16000).input_features

        grbas = torch.tensor(score[0]).unsqueeze(0)
        return asr_mel_feature, output_wavs,mel_specgram, grbas, wavname
