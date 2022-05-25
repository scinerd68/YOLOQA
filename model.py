import torch.nn as nn
from transformers import AutoModelForTokenClassification


class QASpanDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        return outputs


if __name__ == "__main__":
    model = QASpanDetector()
    print(model)