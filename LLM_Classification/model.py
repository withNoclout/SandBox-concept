import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class LLMPreferenceModel(nn.Module):
    """
    The Reward Model.
    
    It uses a pre-trained Transformer backbone (like DeBERTa) and adds a classification head.
    The output is 3 logits: [A wins, B wins, Tie].
    """
    def __init__(self, model_name, num_labels=3):
        super(LLMPreferenceModel, self).__init__()
        
        # Load the configuration
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = num_labels
        
        # Load the pre-trained model with a classification head
        # This handles the pooling and the final linear layer automatically.
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through the model
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return output

# Example usage:
if __name__ == "__main__":
    model_name = "microsoft/deberta-v3-xsmall"
    print(f"Loading model: {model_name}...")
    try:
        model = LLMPreferenceModel(model_name)
        print("Model loaded successfully.")
        
        # Dummy input
        input_ids = torch.randint(0, 1000, (2, 128)) # Batch size 2, seq len 128
        attention_mask = torch.ones((2, 128))
        labels = torch.tensor([0, 1])
        
        print("Running forward pass...")
        output = model(input_ids, attention_mask, labels=labels)
        
        print("Loss:", output.loss.item())
        print("Logits shape:", output.logits.shape) # Should be (2, 3)
        
    except Exception as e:
        print(f"Error: {e}")
