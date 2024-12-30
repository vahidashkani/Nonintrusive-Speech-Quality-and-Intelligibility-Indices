# Nonintrusive-Speech-Quality-and-Intelligibility-Indices
Speech quality and intelligibility are critical in clinical hearing aid (HA) fitting. While validated intrusive predictors like HASPI and HASQI exist, they are not widely implemented in clinical systems. Recent advances in non-intrusive measures, such as those from Clarity Prediction Challenges (CPCs) and HASA-Net, are also not yet accessible to clinicians. However, these advancements rely on datasets from simulated HAs, not the commercial devices used by audiologists. This work aims to develop non-intrusive quality and intelligibility indices using a custom database of noisy speech recorded in a HA test box. In this paper, we introduced a novel non-intrusive model designed to assess speech quality and intelligibility leveraging automatic speech recognition and self-supervised learning techniques. The proposed non-intrusive model was trained to predict the intrusive HASPI/HASQI values, and later validated against subjective intelligibility data obtained from a group of listeners with hearing loss. The proposed model resulted in strong correlations with HASQI (95.96%) and HASPI (97.59%), and a moderate correlation with the subjective intelligibility scores (76.63%).

# Model architecture
The proposed model comprises two main components including feature extraction and downstream modules. The feature extraction component gathers ASR representation, SSL representation, and the Mel-spectrogram features. The downstream modules then map these features to the output labels. An overview of the proposed method is shown in Fig. 1.
In feature extraction module for quality and intelligibility prediction, we use three types of features. The first two are ASR and SSL representations, which are pre-trained using the Whisper and HuBERT models, respectively. Both models consist of 12 transformer encoder layers that demonstrated with 𝑊 and 𝑍.
<img width="815" alt="Screenshot 2024-12-29 at 7 28 42 PM" src="https://github.com/user-attachments/assets/79d9ec2b-963d-4e5d-9940-e05a9471b5aa" />

# Analysis procedure
In our experiments all recordings were resampled to 16000 Hz sample rate. For dataset #1, we computed the HASPI and HASQI values following the procedures outlined by Kates et al. As dataset #2 reports subjective intelligibility scores, we computed the average HASPI value for each list. For the proposed non-intrusive metric, we have extracted the Mel-spectrogram features using 400 FFT filters, and hop length is 320 between STFT windows. In our experiment, the window is Hann with 400 window length, with 40 Mel filter-banks. Fig. 2 represents the Mel-spectrograms of two test samples in dataset #1.
<img width="393" alt="Screenshot 2024-12-29 at 7 31 20 PM" src="https://github.com/user-attachments/assets/a57f605f-723c-4026-a24b-5be068974b63" />

# Results
<img width="410" alt="Screenshot 2024-12-29 at 7 32 37 PM" src="https://github.com/user-attachments/assets/19b36f5e-1b54-42b5-b78e-673ef834a28f" />

