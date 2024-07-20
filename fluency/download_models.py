import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, DistilBertPreTrainedModel, DistilBertModel, DistilBertTokenizer
import torch.nn as nn
import torch

class DistilBertForRegression(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, 1)
        return logits


os.makedirs("./models", exist_ok=True)
os.makedirs("./models/pronunciation_tokenizer", exist_ok=True)
os.makedirs("./models/pronunciation_model", exist_ok=True)
os.makedirs("./models/fluency_tokenizer", exist_ok=True)
os.makedirs("./models/fluency_model", exist_ok=True)

print("Downloading and saving pronunciation tokenizer...")
pronunciation_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
pronunciation_tokenizer.save_pretrained("./models/pronunciation_tokenizer")
print("Pronunciation tokenizer saved successfully.")

print("Downloading and saving pronunciation model...")
pronunciation_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
pronunciation_model.save_pretrained("./models/pronunciation_model")
print("Pronunciation model saved successfully.")

print("Downloading and saving fluency tokenizer...")
fluency_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
fluency_tokenizer.save_pretrained("./models/fluency_tokenizer")
print("Fluency tokenizer saved successfully.")

print("Downloading and saving fluency model...")
fluency_model = DistilBertForRegression.from_pretrained("Kartikeyssj2/Fluency_Scoring_V2")
fluency_model.save_pretrained("./models/fluency_model")
print("Fluency model saved successfully.")

print("Download and save process completed.")