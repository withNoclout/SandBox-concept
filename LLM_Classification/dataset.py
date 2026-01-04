import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class LLMPreferenceDataset(Dataset):
    """
    Custom Dataset for LLM Preference Prediction.
    
    This class takes the raw dataframe and converts it into inputs for the model.
    It combines the Prompt, Response A, and Response B into a single string.
    """
    def __init__(self, df, tokenizer: PreTrainedTokenizer, max_length=1024, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
        # Pre-tokenize the data to save time during training
        # In a real large-scale scenario, we might do this on the fly or cache it.
        self.texts = []
        self.labels = []
        
        for _, row in df.iterrows():
            # Construct the input text
            # Format: [CLS] Prompt [SEP] Response A [SEP] Response B [SEP]
            # Note: The tokenizer will handle [CLS] and [SEP] automatically if we use special functions,
            # but here we construct the string explicitly for clarity or use tokenizer's pair handling.
            # A common approach for 3 segments is: Prompt + Response A + Response B
            
            # We will use a simple concatenation with the tokenizer's separator.
            # DeBERTa uses [SEP] to separate segments.
            
            prompt = str(row['prompt'])
            res_a = str(row['response_a'])
            res_b = str(row['response_b'])
            
            # Option 1: Prompt + Response A + Response B
            text = f"{prompt} {tokenizer.sep_token} {res_a} {tokenizer.sep_token} {res_b}"
            self.texts.append(text)
            
            if self.is_train:
                # Target: [1, 0, 0] for A wins, [0, 1, 0] for B wins, [0, 0, 1] for Tie
                if row['winner_model_a'] == 1:
                    label = 0 # Class 0: A wins
                elif row['winner_model_b'] == 1:
                    label = 1 # Class 1: B wins
                else:
                    label = 2 # Class 2: Tie
                self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length', # Pad to max_length for consistent batch sizes
            return_tensors='pt'   # Return PyTorch tensors
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.is_train:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return item

# Example usage (for testing/learning):
if __name__ == "__main__":
    import pandas as pd
    from transformers import AutoTokenizer
    
    # Create dummy data
    data = {
        'prompt': ['What is AI?', 'Hello'],
        'response_a': ['AI is Artificial Intelligence.', 'Hi there!'],
        'response_b': ['AI is a movie.', 'Hello.'],
        'winner_model_a': [1, 0],
        'winner_model_b': [0, 0],
        'winner_tie': [0, 1]
    }
    df = pd.DataFrame(data)
    
    # Load tokenizer (using a small one for speed)
    model_name = "microsoft/deberta-v3-xsmall" 
    print(f"Loading tokenizer: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        dataset = LLMPreferenceDataset(df, tokenizer, max_length=128)
        print(f"Dataset created with {len(dataset)} samples.")
        
        sample = dataset[0]
        print("\nSample 0:")
        print("Input IDs shape:", sample['input_ids'].shape)
        print("Labels:", sample['labels'])
        print("Decoded Input:", tokenizer.decode(sample['input_ids'], skip_special_tokens=False))
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure you have internet access to download the tokenizer.")
