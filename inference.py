import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle
import os
from model import GraphSmile
from trainer import seed_everything
from dataloader import (
    IEMOCAPDataset_BERT,
    IEMOCAPDataset_BERT4,
    MELDDataset_BERT,
    CMUMOSEIDataset7,
)
from sklearn.metrics import accuracy_score, f1_score

# Define paths (copied from run.py)
MELD_path = 'Dataset/CFN-ESA/meld_multi_features.pkl'
IEMOCAP_path = 'Dataset/CFN-ESA/iemocap_multi_features.pkl'
IEMOCAP4_path = 'Dataset/CFN-ESA/iemocap_multi_features_4.pkl'
CMUMOSEI7_path = 'Dataset/CFN-ESA/cmumosei_multi_regression_features.pkl'

def get_args():
    parser = argparse.ArgumentParser()
    # Arguments from run.py needed for model initialization
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ids') # changed default to 0
    parser.add_argument('--classify', default='emotion', help='sentiment, emotion')
    parser.add_argument('--modals', default='avl', help='modals')
    parser.add_argument('--dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--textf_mode', default='textf0', help='concat4/concat2/textf0/textf1/textf2/textf3/sum2/sum4')
    parser.add_argument('--conv_fpo', nargs='+', type=int, default=[3, 1, 1], help='n_filter,n_padding')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
    parser.add_argument('--win', nargs='+', type=int, default=[17, 17], help='[win_p, win_f]')
    parser.add_argument('--heter_n_layers', nargs='+', type=int, default=[6, 6, 6], help='heter_n_layers')
    parser.add_argument('--drop', type=float, default=0.3, metavar='dropout', help='dropout rate')
    parser.add_argument('--shift_win', type=int, default=12, help='windows of sentiment shift')
    
    # Inference specific arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model .pt file')
    parser.add_argument('--vid', type=str, default=None, help='Video ID to run inference on. If None, prints available VIDs.')
    parser.add_argument('--output_mode', type=str, default='emotion', choices=['both', 'emotion', 'sentiment'], help='Output mode: both, emotion, or sentiment')
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")

    # Determine embedding dims and num classes
    if args.dataset == 'IEMOCAP':
        embedding_dims = [1024, 342, 1582]
        n_classes_emo = 6
        dataset_class = IEMOCAPDataset_BERT
        data_path = IEMOCAP_path
    elif args.dataset == 'IEMOCAP4':
        embedding_dims = [1024, 512, 100]
        n_classes_emo = 4
        dataset_class = IEMOCAPDataset_BERT4
        data_path = IEMOCAP4_path
    elif args.dataset == 'MELD':
        embedding_dims = [1024, 342, 300]
        n_classes_emo = 7
        dataset_class = MELDDataset_BERT
        data_path = MELD_path
    elif args.dataset == 'CMUMOSEI7':
        embedding_dims = [1024, 35, 384]
        n_classes_emo = 7
        dataset_class = CMUMOSEIDataset7
        data_path = CMUMOSEI7_path
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Initialize model
    seed_everything()
    model = GraphSmile(args, embedding_dims, n_classes_emo)
    model.to(device)
    model.eval()

    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    state_dict = torch.load(args.model_path, map_location=device)
    
    # Handle DDP state dict (remove 'module.' prefix)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    print("Model loaded (strict=False) to handle potential dynamic parameters like edge_weight.")

    # Load dataset to find the sample
    print(f"Loading dataset from {data_path}...")
    
    # Try finding in Test set first, then Train set
    found = False
    target_vid = args.vid
    dataset = None
    index = -1
    final_vid = None
    
    # Load dataset (train=False)
    test_set = dataset_class(data_path, train=False)
    # Check if keys are numpy array and convert if needed, else use as is
    test_keys = test_set.keys.tolist() if isinstance(test_set.keys, np.ndarray) else test_set.keys
    
    if target_vid is None:
        print("No video ID provided. Here are some available IDs from the test set:")
        # Print first 10 keys
        print(test_keys[:10])
        return

    if target_vid in test_keys:
        dataset = test_set
        index = test_keys.index(target_vid)
        final_vid = target_vid
        found = True
        print(f"Found {target_vid} in Test set.")
    elif target_vid.isdigit() and int(target_vid) in test_keys:
        dataset = test_set
        index = test_keys.index(int(target_vid))
        final_vid = int(target_vid)
        found = True
        print(f"Found {target_vid} (as int) in Test set.")
    else:
        # Load dataset (train=True)
        train_set = dataset_class(data_path, train=True)
        train_keys = train_set.keys.tolist() if isinstance(train_set.keys, np.ndarray) else train_set.keys
        if target_vid in train_keys:
            dataset = train_set
            index = train_keys.index(target_vid)
            final_vid = target_vid
            found = True
            print(f"Found {target_vid} in Train set.")
        # Try int check for train set as well if needed
        elif target_vid.isdigit() and int(target_vid) in train_keys:
            dataset = train_set
            index = train_keys.index(int(target_vid))
            final_vid = int(target_vid)
            found = True
            print(f"Found {target_vid} (as int) in Train set.")

    if not found:
        print(f"Video ID {target_vid} not found in {args.dataset}.")
        return

    # Get sample
    
    # Retrieve sentences using final_vid
    try:
        sentences = dataset.videoSentence[final_vid]
    except Exception as e:
        print(f"Warning: Could not retrieve sentences for {final_vid}. Error: {e}")
        sentences = []

    print("-" * 50)
    print(f"Total Utterances: {len(sentences)}")
    for i, s in enumerate(sentences):
        print(f"Index {i}: {s}")
    print("-" * 50)
    # The dataset __getitem__ returns a tuple where the last element is VID (which we know), preventing us from using collate_fn directly easily if we just fetch one item.
    # However, model expects batched input. We should use collate_fn.
    
    sample = dataset[index]
    # Collate fn expects a list of samples
    batch = dataset.collate_fn([sample])
    
    # Unpack batch
    # dataloader.py colate_fn returns:
    # [ (pad_sequence(...) if ... else ... ) for i in dat ]
    # Essentially it returns a list corresponding to the columns of the dataframe created from data.
    
    # In train_or_eval_model:
    # textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, label_emotion, label_sentiment = ([d.cuda() for d in data[:-1]] if cuda else data[:-1])
    
    # Let's match this unpacking
    if cuda:
        batch_data = [d.cuda() for d in batch[:-1]]
    else:
        batch_data = batch[:-1]

    textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, label_emotion, label_sentiment = batch_data

    # Mapping (based on trainer.py and dataloader.py):
    # textf0..3 (0-3), visuf (4), acouf (5)
    # qmask (6) -> speakers
    # umask (7) -> length mask
    
    # Calculate dialogue lengths using umask (index 7) which contains 1s for the sequence length
    dia_lengths = []
    for j in range(umask.size(1)): # batch size 1
        dia_lengths.append((umask[:, j] == 1).nonzero().tolist()[-1][0] + 1)
    
    with torch.no_grad():
        # Model forward expects: textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask, dia_lengths
        # Note: umask (length mask) is passed before qmask (speaker mask) in the model signature
        logit_emo, logit_sen, logit_sft, _ = model(
            textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask, dia_lengths
        )
        
        prob_emo = F.softmax(logit_emo, -1) # Use softmax for probabilities
        pred_emo = torch.argmax(prob_emo, dim=1).cpu().numpy()
        
        prob_sen = F.softmax(logit_sen, -1)
        pred_sen = torch.argmax(prob_sen, dim=1).cpu().numpy()

    # Print results
    print("-" * 50)
    print(f"Inference Results for VID: {target_vid}")
    print("-" * 50)
    print(f"Emotion Logits: {logit_emo.cpu().numpy()}")
    print(f"Sentiment Logits: {logit_sen.cpu().numpy()}")
    print("-" * 20)
    
    # Map predictions to labels (if available)
    # IEMOCAP Labels: {0: 'hap', 1: 'sad', 2: 'neu', 3: 'ang', 4: 'exc', 5: 'fru'} (Check dataloader or paper for precise mapping)
    # run.py says:
    # 1, 3, 5 -> 0 (Negative?)
    # 2 -> 1 (Neutral?)
    # 0, 4 -> 2 (Positive?)
    # wait, run.py line 108:
    # if e in [1, 3, 5]: array.append(0)
    # elif e == 2: array.append(1)
    # elif e in [0, 4]: array.append(2)
    
    # For IEMOCAP, typically:
    # 0: happy, 1: sad, 2: neutral, 3: angry, 4: excited, 5: frustrated
    # Sentiment usually: 0: negative, 1: neutral, 2: positive
    


    # Prepare Label Map
    emotion_map = None
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'} # Common for most datasets here

    if args.dataset == 'MELD':
        emotion_map = {
            0: 'Neutral', 1: 'Surprise', 2: 'Fear', 3: 'Sadness', 
            4: 'Joy', 5: 'Disgust', 6: 'Anger'
        }
    elif args.dataset == 'IEMOCAP': 
        # For IEMOCAP: 0:hap, 1:sad, 2:neu, 3:ang, 4:exc, 5:fru
        emotion_map = {
            0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry', 
            4: 'Excited', 5: 'Frustrated'
        }
    elif args.dataset == 'IEMOCAP4':
        # Inferred from dataloader sentiment logic: 
        # 0->Pos, 2->Neu, 1,3->Neg
        emotion_map = {
            0: 'Happy', 1: 'Sad', 2: 'Neutral', 3: 'Angry'
        }

    # Calculate metrics
    def print_mismatches(pred, truth, label_name, sentences=None, label_map=None):
        print("-" * 20)
        print(f"{label_name} Mismatches:")
        mismatches = []
        for i, (p, t) in enumerate(zip(pred, truth)):
            if p != t:
                p_label = f" ({label_map[p]})" if label_map and p in label_map else ""
                t_label = f" ({label_map[t]})" if label_map and t in label_map else ""
                
                text = f" | Text: \"{sentences[i]}\"" if sentences and i < len(sentences) else ""
                mismatches.append(f"Index {i}: Pred {p}{p_label} != Truth {t}{t_label}{text}")
        
        if not mismatches:
            print("None. Perfect Match!")
        else:
            for m in mismatches:
                print(m)
        
        acc = accuracy_score(truth, pred)
        f1 = f1_score(truth, pred, average='weighted')
        print("-" * 20)
        print(f"{label_name} Accuracy: {acc:.4f}")
        print(f"{label_name} F1 Score: {f1:.4f}")

    # Map predictions to labels (if available)
    label_emotion_np = label_emotion.cpu().numpy().flatten()
    label_sentiment_np = label_sentiment.cpu().numpy().flatten()
    
    if args.output_mode in ['both', 'emotion']:
        print(f"Predicted Emotion Index: {pred_emo}")
        print(f"Ground Truth Emotion: {label_emotion_np}")
        print_mismatches(pred_emo, label_emotion_np, "Emotion", sentences, emotion_map)
    
    if args.output_mode in ['both', 'sentiment']:
        print("-" * 20)
        print(f"Predicted Sentiment Index: {pred_sen}")
        print(f"Ground Truth Sentiment: {label_sentiment_np}")
        print_mismatches(pred_sen, label_sentiment_np, "Sentiment", sentences, sentiment_map)
    
    print("-" * 50)

if __name__ == '__main__':
    main()
