
import pandas as pd
import os

base_path = '/mnt/Work/ML/Dataset/MELD.Raw'
train_csv = os.path.join(base_path, 'train/train_sent_emo.csv')
dev_csv = os.path.join(base_path, 'dev_sent_emo.csv')
test_csv = os.path.join(base_path, 'test_sent_emo.csv')

def count_unique_dialogues(csv_path, split_name):
    if not os.path.exists(csv_path):
        print(f"{split_name} CSV not found at {csv_path}")
        return 0, set()
    
    try:
        df = pd.read_csv(csv_path)
        # Check column name for Dialogue_ID
        if 'Dialogue_ID' in df.columns:
            unique_ids = df['Dialogue_ID'].unique()
            count = len(unique_ids)
            print(f"{split_name}: {count} unique dialogues. (Min: {min(unique_ids)}, Max: {max(unique_ids)})")
            return count, set(unique_ids)
        else:
            print(f"Dialogue_ID column not found in {split_name}")
            return 0, set()
    except Exception as e:
        print(f"Error reading {split_name}: {e}")
        return 0, set()

train_count, _ = count_unique_dialogues(train_csv, "Train")
dev_count, _ = count_unique_dialogues(dev_csv, "Dev")
test_count, _ = count_unique_dialogues(test_csv, "Test")

print("-" * 20)
print(f"Total Train + Dev Dialogues: {train_count + dev_count}")
