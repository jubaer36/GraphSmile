
import pickle
import numpy as np
from dataloader import MELDDataset_BERT

MEL_PATH = 'Dataset/CFN-ESA/meld_multi_features.pkl'

def inspect_meld():
    print(f"Loading MELD from {MEL_PATH}...")
    try:
        # Load directly to inspect keys if possible, or use the class
        dataset = MELDDataset_BERT(MEL_PATH, train=False)
        print(f"Dataset loaded. Number of keys: {len(dataset.keys)}")
        
        # Check type of keys
        first_key = dataset.keys[0]
        print(f"Type of keys: {type(first_key)}")
        print(f"First 10 keys: {dataset.keys[:10]}")
        
        # Check if keys match 'diaX' pattern logic
        # Let's inspect content of key 0 (if valid) or first valid key
        key_to_inspect = 0 if 0 in dataset.keys else dataset.keys[0]
        
        print(f"\nInspecting Key: {key_to_inspect}")
        sentences = dataset.videoSentence[key_to_inspect]
        print(f"Number of sentences/utterances: {len(sentences)}")
        for i, s in enumerate(sentences):
            print(f"  Utt {i}: {s}")

        # Check for key 100
        if 100 in dataset.keys:
            print(f"\nFound Key: 100")
            sentences_100 = dataset.videoSentence[100]
            print(f"Number of sentences/utterances in dia100: {len(sentences_100)}")
            print(f"  First utterance: {sentences_100[0]}")
        else:
            print("\nKey 100 not found in Test set (might be in Train).")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_meld()
