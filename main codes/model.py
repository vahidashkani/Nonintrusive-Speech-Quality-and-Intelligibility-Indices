import torch
import torch.nn as nn

class GRBASPredictor(nn.Module):
    # ssl_model: HuBERT
    # asr_model: Whisper
    def __init__(self, ssl_model, asr_model, decoder_input_ids, ssl_out_dim, grbas_dim):
        super(GRBASPredictor, self).__init__()
        self.asr_model = asr_model
        self.ssl_model = ssl_model
        self.decoder_input_ids = decoder_input_ids
        self.bottleneck_dim = 120
        self.features_out = ssl_out_dim  # 768
        self.grbas_dim = grbas_dim
        self.downstream_layer_1 = nn.LSTM(360, 16, 2, bidirectional=True)
        self.downstream_layer_2 = nn.Linear(32, 1, bias=False)  # Adjust this line
    
        self.ssl_weight = nn.Parameter(torch.rand(6), requires_grad=True)
        self.asr_weight = nn.Parameter(torch.rand(6), requires_grad=True)
    
        # 12 adapters, first 6 for ssl features, last 6 for asr features
        self.adapter_1 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_2 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_3 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_4 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_5 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_6 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_7 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_8 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_9 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_10 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_11 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
        self.adapter_12 = nn.Sequential(nn.Linear(self.features_out, self.bottleneck_dim), nn.LeakyReLU(0.05), nn.LayerNorm(self.bottleneck_dim))
    
    def forward(self, asr_mel_feature, wav, mel_features):
        # mel_spectrogram
        mel_features = mel_features.reshape(1, -1, 120)
        #print("mel_features shape:", mel_features.shape)
        
        # ssl_features
        wav = wav.squeeze(1)  # batch x audio_len
        outputs = self.ssl_model(wav, output_hidden_states=True)
        ssl_hidden_state = outputs.hidden_states
        #print("ssl_hidden_state shape:", ssl_hidden_state[-1].shape)
        
        # asr_features
        x = self.asr_model(asr_mel_feature.squeeze(0), decoder_input_ids=self.decoder_input_ids, output_hidden_states=True)
        asr_hidden_state = x.encoder_hidden_states
        #print("asr_hidden_state shape:", asr_hidden_state[-1].shape)
        
        hidden_state_1 = self.adapter_1(ssl_hidden_state[-1])
        hidden_state_2 = self.adapter_2(ssl_hidden_state[-2])
        hidden_state_3 = self.adapter_3(ssl_hidden_state[-3])
        hidden_state_4 = self.adapter_4(ssl_hidden_state[-4])
        hidden_state_5 = self.adapter_5(ssl_hidden_state[-5])
        hidden_state_6 = self.adapter_6(ssl_hidden_state[-6])
        
        hidden_state_7 = self.adapter_7(asr_hidden_state[-1])
        hidden_state_8 = self.adapter_8(asr_hidden_state[-2])
        hidden_state_9 = self.adapter_9(asr_hidden_state[-3])
        hidden_state_10 = self.adapter_10(asr_hidden_state[-4])
        hidden_state_11 = self.adapter_11(asr_hidden_state[-5])
        hidden_state_12 = self.adapter_12(asr_hidden_state[-6])
    
        ssl_new_state = torch.stack([hidden_state_1, hidden_state_2, hidden_state_3, hidden_state_4, hidden_state_5, hidden_state_6], dim=-1)
        asr_new_state = torch.stack([hidden_state_7, hidden_state_8, hidden_state_9, hidden_state_10, hidden_state_11, hidden_state_12], dim=-1)
    
        #print("ssl_new_state shape:", ssl_new_state.shape)
        #print("asr_new_state shape:", asr_new_state.shape)
        
        ssl_weight = self.ssl_weight.unsqueeze(-1)
        ssl_weight = nn.functional.softmax(ssl_weight, dim=0)
        ssl_x = torch.matmul(ssl_new_state, ssl_weight).squeeze(-1)
        
        asr_weight = self.asr_weight.unsqueeze(-1)
        asr_weight = nn.functional.softmax(asr_weight, dim=0)
        asr_x = torch.matmul(asr_new_state, asr_weight).squeeze(-1)
        #print("ssl_x shape:", ssl_x.shape)
        #print("asr_x shape:", asr_x.shape)
        
        _, T, _ = mel_features.shape
    
        # align
        all_features = torch.cat((asr_x[:, :T, :], mel_features, ssl_x), dim=-1)
        #print("all_features shape before LSTM:", all_features.shape)
        
        # LSTM expects input size of 360
        all_features, _ = self.downstream_layer_1(all_features)
        # classification
        all_features = self.downstream_layer_2(all_features)  # Apply final Linear layer
        #print("all_features shape after LSTM:", all_features.shape)
        
        all_features = torch.mean(all_features, dim=1)
        #print("all_features shape after mean:", all_features.shape)
        
        
        #print("all_features shape after classification:", all_features.shape)
        return all_features.squeeze(1)
