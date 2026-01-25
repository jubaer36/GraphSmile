# GraphSmile
The official implementation of the paper "[Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition](https://doi.org/10.1109/TPAMI.2025.3581236)", which has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).  
Authors: Jiang Li, Xiaoping Wang, Zhigang Zeng  
Affiliation: Huazhong University of Science and Technology (HUST)  

## Citation
```bibtex
@article{li2025tracing,
    title={Tracing intricate cues in dialogue: Joint graph structure and sentiment dynamics for multimodal emotion recognition},
    author={Jiang Li and Xiaoping Wang and Zhigang Zeng},
    year={2025},
    journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume={47},
    number={10},
    pages = {8786-8803},
    doi={10.1109/TPAMI.2025.3581236}
}
```

## Requirement
Checking and installing environmental requirements
```python
pip install -r requirements.txt
```
## Datasets
链接: https://pan.baidu.com/s/1u1efdbBV3HP8FLj3Gy1bvQ
提取码: ipnv
Google Drive: https://drive.google.com/drive/folders/1l_ex1wnAAMpEtO71rjjM1MKC7W_olEVi?usp=drive_link

Adding the dataset path to the corresponding location in the run.py file, e.g. IEMOCAP_path = "".

## Run
### IEMOCAP-6
```bash
python -u run.py --gpu 0 --port 1530 --classify emotion \
--dataset IEMOCAP --epochs 120 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 1e-04 --batch_size 16 --hidden_dim 512 \
--win 17 17 --heter_n_layers 7 7 7 --drop 0.2 --shift_win 19 --lambd 1.0 1.0 0.7
```

### IEMOCAP-4
```bash
python -u run.py --gpu 0 --port 1531 --classify emotion \
--dataset IEMOCAP4 --epochs 120 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 3e-04 --batch_size 16 --hidden_dim 256 \
--win 5 5 --heter_n_layers 4 4 4 --drop 0.2 --shift_win 10 --lambd 1.0 0.6 0.6
```

### MELD
```bash
python -u run.py --gpu 0 --port 1532 --classify emotion \
--dataset MELD --epochs 50 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 7e-05 --batch_size 16 --hidden_dim 384 \
--win 3 3 --heter_n_layers 5 5 5 --drop 0.2 --shift_win 3 --lambd 1.0 0.5 0.2
```

### CMUMOSEI
```bash
python -u run.py --gpu 0 --port 1534 --classify emotion \
--dataset CMUMOSEI7 --epochs 60 --textf_mode textf0 \
--loss_type emo_sen_sft --lr 8e-05 --batch_size 32 --hidden_dim 256 \
--win 5 5 --heter_n_layers 2 2 2 --drop 0.4 --shift_win 2 --lambd 1.0 0.8 1.0
```

## Run Inference
To run inference on specific list of samples using a trained model, use `inference.py`.

### Arguments
- `--model_path`: Path to the saved model `.pt` file (e.g., `saved_models/avl_IEMOCAP_best.pt`).
- `--vid`: (Optional) Valid Video ID from the dataset. If omitted, the script lists available VIDs.
- `--output_mode`: (Optional) Choose output to display: `'emotion'` (default), `'sentiment'`, or `'both'`.
- **Hyperparameters**: You **MUST** provide the same model hyperparameters used during training (e.g., `--hidden_dim`, `--heter_n_layers`).

### Example: IEMOCAP
```bash
python inference.py \
    --model_path saved_models/avl_IEMOCAP_best.pt \
    --dataset IEMOCAP \
    --gpu 0 \
    --hidden_dim 512 \
    --heter_n_layers 7 7 7 \
    --vid Ses05F_impro08 \
    --output_mode emotion
```

### Example: IEMOCAP4
```bash
python inference.py \
    --model_path saved_models/avl_IEMOCAP4_best.pt \
    --dataset IEMOCAP4 \
    --gpu 0 \
    --hidden_dim 256 \
    --heter_n_layers 4 4 4 \
    --vid Ses05M_impro02 \
    --output_mode emotion
```

### Example: MELD
```bash
python inference.py \
    --model_path saved_models/avl_MELD_best.pt \
    --dataset MELD \
    --gpu 0 \
    --hidden_dim 384 \
    --heter_n_layers 5 5 5 \
    --vid 1153 \
    --output_mode emotion
```

### Example: CMUMOSEI7
```bash
python inference.py \
    --model_path saved_models/avl_CMUMOSEI7_best.pt \
    --dataset CMUMOSEI7 \
    --gpu 0 \
    --hidden_dim 256 \
    --heter_n_layers 2 2 2 \
    --vid k8yDywC4gt8 \
    --output_mode emotion
```