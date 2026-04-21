import torch
import torch.nn as nn

from models.FuzzyCDM import LLM, DINA, DINO, NeuralCD


class VanillaCDM(nn.Module):
    '''Plain CDM with no process data component.'''

    def __init__(self, student_n, exer_n, knowledge_n, Q, cdm="DINA"):
        super(VanillaCDM, self).__init__()
        self.Q = Q
        self.cdm_type = cdm

        if cdm == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif cdm == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif cdm == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif cdm == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, log_seq=None):
        return self.cdm(stu_id, exer_id), None

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMFlat(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, proc_dim, Q, cdm="LLM", hidden_dim=30):
        super(ProcCDMFlat, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.proc_dim = proc_dim
        self.Q = Q
        self.cdm_type = cdm

        if self.cdm_type == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        self.proc_layers = nn.Sequential(
            nn.Linear(self.stu_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, self.proc_dim)
        )   
    
    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id):
        output = self.cdm(stu_id, exer_id)
        output2 = self.proc_layers(self.get_knowledge_status(stu_id))
        return output, output2
    
    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMFlatV2(nn.Module):
    '''
    CDM whose θ is refined by flat process features (log RT, nactions) via backprop.

    A process encoder maps cat(proc_features, θ) → θ_proc (a process-informed
    knowledge estimate). An MSE loss between θ_proc and the CDM student embedding
    pulls θ toward what process data suggests, without modifying θ at inference time.

    Training loss: L_BCE(CDM(θ, item), y) + λ · MSE(θ_proc, θ)
    Inference:     standard CDM(θ, item) — no process data needed.
    '''

    def __init__(self, student_n, exer_n, knowledge_n, n_proc, Q,
                 cdm="LLM", hidden_dim=30):
        super(ProcCDMFlatV2, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.Q = Q
        self.cdm_type = cdm

        if cdm == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif cdm == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif cdm == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif cdm == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        # Process encoder: proc_features → θ_proc in [0, 1]^K
        self.proc_encoder = nn.Sequential(
            nn.Linear(n_proc, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, knowledge_n),
            nn.Sigmoid(),
        )

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, proc_features=None):
        '''
        proc_features: (B, n_proc) — only used during training.
        Returns (pred, theta_proc) during training, (pred, None) at inference.
        theta_proc is used by the caller to compute MSE(theta_proc, θ).
        '''
        pred = self.cdm(stu_id, exer_id)
        if proc_features is not None and self.training:
            theta_proc = self.proc_encoder(proc_features)          # (B, K)
        else:
            theta_proc = None
        return pred, theta_proc

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMFlatV3(nn.Module):
    '''
    CDM with direct logit-space θ adjustment using flat process features.

    At every forward pass:
      1. enc_input = cat(proc_features, θ)       (B, n_proc + K)
      2. Δ = delta_fc(enc_input)                  (B, K)
      3. g = gate(enc_input)                      (B, K), sigmoid
      4. θ_aug = sigmoid(logit(θ) + g ⊙ Δ)
      5. pred = cdm.response(θ_aug, exer_id)

    No auxiliary loss — only BCE. θ is adjusted at every inference call.
    '''

    def __init__(self, student_n, exer_n, knowledge_n, n_proc, Q,
                 cdm="LLM", hidden_dim=30):
        super(ProcCDMFlatV3, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.Q = Q
        self.cdm_type = cdm

        if cdm == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif cdm == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif cdm == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif cdm == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        in_dim = n_proc + knowledge_n

        self.delta_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, knowledge_n),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, knowledge_n), nn.Sigmoid(),
        )

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, proc_features):
        theta = torch.sigmoid(self.cdm.student_emb[stu_id])        # (B, K)
        enc   = torch.cat([proc_features, theta], dim=-1)          # (B, n_proc+K)
        delta = self.delta_fc(enc)                                  # (B, K)
        g     = self.gate(enc)                                      # (B, K)
        theta_aug = torch.sigmoid(
            torch.logit(theta.clamp(1e-6, 1 - 1e-6)) + g * delta
        )
        return self.cdm.response(theta_aug, exer_id), None

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMAttn(nn.Module):
    '''
    CDM whose θ is refined by log-sequence process data via a Transformer encoder,
    following the same training objective as ProcCDMLSTMV2.

    A Transformer encodes the full log sequence; mean pooling over non-padding
    positions produces a summary vector that is projected to θ_proc in [0,1]^K.
    An MSE loss between θ_proc and the CDM student embedding pulls θ toward what
    the process data suggests, without modifying θ at inference time.

    Training loss: L_BCE(CDM(θ, item), y) + λ · MSE(θ_proc, θ)
    Inference:     standard CDM(θ, item) — no log sequence needed.

    Differs from LSTMV2 in using multi-head self-attention (non-causal, full
    sequence summarisation) rather than an LSTM final hidden state.
    '''
    LOG_INPUT_DIM = 5  # 4-dim event embedding + 1 timestamp

    def __init__(self, student_n, exer_n, knowledge_n, proc_dim, Q,
                 cdm="LLM", n_heads=4, n_layers=2, dropout=0.1, hidden_dim=30):
        super(ProcCDMAttn, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = knowledge_n
        self.proc_dim = proc_dim
        self.Q = Q
        self.cdm_type = cdm

        if self.cdm_type == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        # Project raw log features -> proc_dim for the Transformer
        self.input_proj = nn.Linear(self.LOG_INPUT_DIM, proc_dim)

        # Non-causal Transformer encoder: attends over the full sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proc_dim,
            nhead=n_heads,
            dim_feedforward=proc_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projects mean-pooled Transformer output -> θ_proc in [0, 1]^K
        self.proc_fc = nn.Sequential(
            nn.Linear(proc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, knowledge_n),
            nn.Sigmoid(),
        )

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, log_seq=None):
        """
        log_seq: PackedSequence or padded (B, T, 5) — only used during training.
        Returns (pred, theta_proc) during training, (pred, None) at inference.
        theta_proc is used by the caller to compute MSE(theta_proc, θ).
        """
        pred = self.cdm(stu_id, exer_id)

        if log_seq is not None and self.training:
            if isinstance(log_seq, nn.utils.rnn.PackedSequence):
                padded, lengths = nn.utils.rnn.pad_packed_sequence(log_seq, batch_first=True)
            else:
                padded = log_seq
                lengths = None

            x = self.input_proj(padded)  # (B, T, proc_dim)

            # Key-padding mask: True at padding positions (ignored in attention)
            if lengths is not None:
                T = padded.size(1)
                key_padding_mask = (
                    torch.arange(T, device=x.device).unsqueeze(0)
                    >= lengths.to(x.device).unsqueeze(1)
                )  # (B, T)
            else:
                key_padding_mask = None

            enc = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, T, proc_dim)

            # Mean-pool over non-padding positions
            if key_padding_mask is not None:
                valid = (~key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
                summary = (enc * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            else:
                summary = enc.mean(dim=1)  # (B, proc_dim)

            theta_proc = self.proc_fc(summary)  # (B, K)
        else:
            theta_proc = None

        return pred, theta_proc

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMLSTM(nn.Module):
    '''
    NeuralCDM with LSTM-based process feature extraction.
    log_seq: (batch, T, 5) padded tensor of log events,
             each step is [emb_0..emb_3, timestamp].
    '''
    LOG_INPUT_DIM = 5  # 4-dim event embedding + 1 timestamp

    def __init__(self, student_n, exer_n, knowledge_n, proc_dim, Q, n_event_types, cdm="LLM"):
        super(ProcCDMLSTM, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.proc_dim = proc_dim
        self.Q = Q
        self.cdm_type = cdm

        if self.cdm_type == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        self.proc_lstm = nn.LSTM(
            input_size=self.LOG_INPUT_DIM + self.knowledge_dim,
            hidden_size=proc_dim,
            batch_first=True,
        )
        # Next action-type prediction head: h_t -> logits over event vocabulary
        self.proc_fc = nn.Linear(proc_dim, n_event_types)

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, log_seq):
        """
        log_seq: padded tensor (batch, T, 5) or PackedSequence.
        At every LSTM step the student knowledge profile (knowledge_n) is
        concatenated with the log features, giving input_size = 5 + knowledge_n.
        Returns:
            output      — CDM prediction (batch,)
            lstm_out    — all LSTM step outputs, padded: (batch, T_max, proc_dim)
        Use proc_fc(lstm_out[:, :-1, :]) vs event_idxs[:, 1:] for next-event loss.
        """
        output = self.cdm(stu_id, exer_id)

        # stu_profile: (batch, knowledge_n) — current knowledge state per student
        stu_profile = self.cdm.profile()[stu_id]

        if isinstance(log_seq, nn.utils.rnn.PackedSequence):
            padded, lengths = nn.utils.rnn.pad_packed_sequence(log_seq, batch_first=True)
            profile_exp = stu_profile.unsqueeze(1).expand(-1, padded.size(1), -1)
            padded = torch.cat([padded, profile_exp], dim=-1)
            log_seq_aug = nn.utils.rnn.pack_padded_sequence(
                padded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.proc_lstm(log_seq_aug)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            profile_exp = stu_profile.unsqueeze(1).expand(-1, log_seq.size(1), -1)
            log_seq_aug = torch.cat([log_seq, profile_exp], dim=-1)
            lstm_out, _ = self.proc_lstm(log_seq_aug)

        # lstm_out: (batch, T_max, proc_dim)
        return output, lstm_out

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMLSTMV2(nn.Module):
    '''
    CDM whose θ is refined by log-sequence process data via backprop.

    An LSTM encodes the log sequence into a summary vector, which is projected
    to θ_proc (a process-informed knowledge estimate in [0,1]^K). An MSE loss
    between θ_proc and the CDM student embedding pulls θ toward what the
    process data suggests, without modifying θ at inference time.

    Training loss: L_BCE(CDM(θ, item), y) + λ · MSE(θ_proc, θ)
    Inference:     standard CDM(θ, item) — no log sequence needed.
    '''
    LOG_INPUT_DIM = 5  # 4-dim event embedding + 1 timestamp

    def __init__(self, student_n, exer_n, knowledge_n, proc_dim, Q, cdm="LLM", hidden_dim=30):
        super(ProcCDMLSTMV2, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = knowledge_n
        self.proc_dim = proc_dim
        self.Q = Q
        self.cdm_type = cdm

        if self.cdm_type == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        self.proc_lstm = nn.LSTM(
            input_size=self.LOG_INPUT_DIM,
            hidden_size=proc_dim,
            batch_first=True,
        )

        # Projects LSTM final hidden state → θ_proc in [0, 1]^K
        self.proc_fc = nn.Sequential(
            nn.Linear(proc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, knowledge_n),
            nn.Sigmoid(),
        )

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, log_seq=None):
        """
        log_seq: PackedSequence or padded (B, T, 5) — only used during training.
        Returns (pred, theta_proc) during training, (pred, None) at inference.
        theta_proc is used by the caller to compute MSE(theta_proc, θ).
        """
        pred = self.cdm(stu_id, exer_id)

        if log_seq is not None and self.training:
            if isinstance(log_seq, nn.utils.rnn.PackedSequence):
                _, (h_n, _) = self.proc_lstm(log_seq)
            else:
                _, (h_n, _) = self.proc_lstm(log_seq)

            theta_proc = self.proc_fc(h_n.squeeze(0))  # (B, K)
        else:
            theta_proc = None

        return pred, theta_proc

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()


class ProcCDMLSTMV3(nn.Module):
    '''
    CDM with direct logit-space θ adjustment using LSTM-encoded process data.

    At every forward pass:
      1. LSTM encodes log_seq (augmented with θ) → h_final
      2. enc = cat(h_final, θ)
      3. Δ = delta_fc(enc)        (B, K)
      4. g = gate(enc)            (B, K), sigmoid
      5. θ_aug = sigmoid(logit(θ) + g ⊙ Δ)
      6. pred = cdm.response(θ_aug, exer_id)

    No auxiliary loss — only BCE. θ is adjusted at every inference call.
    '''
    LOG_INPUT_DIM = 5  # 4-dim event embedding + 1 timestamp

    def __init__(self, student_n, exer_n, knowledge_n, proc_dim, Q,
                 cdm="LLM", hidden_dim=30):
        super(ProcCDMLSTMV3, self).__init__()

        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = knowledge_n
        self.proc_dim = proc_dim
        self.Q = Q
        self.cdm_type = cdm

        if self.cdm_type == "LLM":
            self.cdm = LLM(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINA":
            self.cdm = DINA(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "DINO":
            self.cdm = DINO(student_n, exer_n, knowledge_n, Q)
        elif self.cdm_type == "NeuralCD":
            self.cdm = NeuralCD(student_n, exer_n, knowledge_n, Q)

        self.proc_lstm = nn.LSTM(
            input_size=self.LOG_INPUT_DIM + knowledge_n,
            hidden_size=proc_dim,
            batch_first=True,
        )

        in_dim = proc_dim + knowledge_n
        self.delta_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, knowledge_n),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, knowledge_n), nn.Sigmoid(),
        )

    def apply_clipper(self):
        self.cdm.apply_clipper()

    def forward(self, stu_id, exer_id, log_seq):
        theta = torch.sigmoid(self.cdm.student_emb[stu_id])  # (B, K)

        if isinstance(log_seq, nn.utils.rnn.PackedSequence):
            padded, lengths = nn.utils.rnn.pad_packed_sequence(log_seq, batch_first=True)
            profile_exp = theta.unsqueeze(1).expand(-1, padded.size(1), -1)
            padded_aug = torch.cat([padded, profile_exp], dim=-1)
            packed_aug = nn.utils.rnn.pack_padded_sequence(
                padded_aug, lengths, batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.proc_lstm(packed_aug)
        else:
            profile_exp = theta.unsqueeze(1).expand(-1, log_seq.size(1), -1)
            _, (h_n, _) = self.proc_lstm(torch.cat([log_seq, profile_exp], dim=-1))

        h_final = h_n.squeeze(0)                             # (B, proc_dim)
        enc = torch.cat([h_final, theta], dim=-1)            # (B, proc_dim + K)
        delta = self.delta_fc(enc)                           # (B, K)
        g = self.gate(enc)                                   # (B, K)
        theta_aug = torch.sigmoid(
            torch.logit(theta.clamp(1e-6, 1 - 1e-6)) + g * delta
        )
        return self.cdm.response(theta_aug, exer_id), None

    def profile(self):
        return self.cdm.profile()

    def get_knowledge_status(self, stu_id):
        return self.cdm.get_knowledge_status(stu_id)

    def get_exer_params(self):
        return self.cdm.get_exer_params()
