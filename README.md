# Context-Dependent Diphone Modeling for Intracortical Speech Decoding

Diphone-based decoder for the Brain-to-Text '25 competition. This repository contains the implementation of a context-dependent diphone modeling approach for decoding speech from intracortical brain recordings.

## Key Results

- **Phoneme Error Rate (PER)**: 16.36%
- **Word Error Rate (WER)**: 40%
- **Place Clustering Factor**: 1.70x
- Achieved through context-dependent diphone modeling with articulatory feature integration

## Architecture

- **Model Type**: 5-layer Gated Recurrent Unit (GRU)
- **Hidden Size**: 768 units
- **Output Classes**: 1,601 diphone classes
- **Total Parameters**: ~199M
- **Context**: Phoneme pairs (diphones) with articulatory features

## Repository Structure

```
braintotext/
├── scripts/
│   ├── training/          # Core training scripts
│   │   ├── train_model.py         # Main training script
│   │   ├── rnn_model.py           # GRU model architecture
│   │   ├── rnn_trainer.py         # Training loop and logic
│   │   ├── dataset.py             # Data loading and preprocessing
│   │   └── data_augmentations.py # Data augmentation strategies
│   ├── decoding/          # Inference and submission
│   │   └── generate_submission.py # Generate competition submissions
│   ├── analysis/          # Evaluation and analysis
│   │   ├── evaluate_model.py      # Model evaluation
│   │   ├── evaluate_diphone.py    # Diphone-specific metrics
│   │   ├── generate_phonetics_figures.py  # Visualization
│   │   └── extract_hidden_states_pca.py   # Hidden state analysis
│   └── utils/             # Utility modules
│       ├── phoneme_vocab.py       # Phoneme/diphone vocabulary
│       ├── phoneme_to_features.py # Articulatory features
│       ├── download_data.py       # Data download utilities
│       └── setup.py               # Setup utilities
├── paper/
│   └── figures/           # Paper figures
├── configs/               # Configuration files
├── prelim_code/          # Experimental/preliminary code
└── README.md

```

## Usage

### Training

To train the diphone model:

```bash
cd scripts/training
python train_model.py --config ../../configs/diphone_config.yaml
```

### Evaluation

To evaluate a trained model:

```bash
cd scripts/analysis
python evaluate_model.py --checkpoint <path_to_checkpoint>
```

### Generating Submissions

To generate competition submission files:

```bash
cd scripts/decoding
python generate_submission.py --model <path_to_model> --output <output_path>
```

### Data Download

To download the competition data:

```bash
cd scripts/utils
python download_data.py
```

## Requirements

See `requirements.txt` for dependencies. Key requirements:
- PyTorch >= 2.0
- NumPy, SciPy
- Matplotlib, Pandas
- torchaudio

Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Details

The model uses context-dependent diphone representations, which capture phoneme-to-phoneme transitions and coarticulation effects. Each diphone is augmented with articulatory features (place, manner, voicing) to improve generalization and reduce the effective class space.

### Key Features:
- **Diphone Units**: Models phoneme pairs instead of isolated phonemes
- **Articulatory Features**: Incorporates phonetic features for better generalization
- **Deep GRU Architecture**: 5-layer recurrent network with 768 hidden units
- **Data Augmentation**: Time warping, noise injection, and temporal masking

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Tereza Okalova
[Contact information]

## Acknowledgments

This work was developed for the Brain-to-Text 2025 competition.
