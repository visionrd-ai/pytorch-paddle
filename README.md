# Paddle2PyTorch: Train & Test PaddleOCR models in PyTorch

This repository re-implements PaddleOCR's latest OCR models in PyTorch, removing dependency from Paddle as a framework and providing more configurability and insight into the training pipeline. It aims to recreate the original performance and structure of PaddleOCR models in PyTorch, allowing for easier integration and experimentation within PyTorch-based pipelines. The project includes model configurations, dataset management, and training/evaluation scripts.

## Features

- **Model Implementation**: Transforms PaddleOCR v4 English models into PyTorch, replicating the architecture and hyperparameters.
- **Dual-Head OCR**: Supports models with both CTC and NRTR heads for robust recognition capabilities.
- **Flexible Configurations**: Model, training, and data configurations are separated into JSON files for easy modification.
- **Training & Evaluation Scripts**: Includes scripts to train and evaluate the converted models on custom datasets.

## Setup

### Prerequisites

- **Python 3.7+**
- **PyTorch 1.8+**
- **PaddlePaddle** (for model reference during conversion)
- **Additional Libraries**: Refer to `requirements.txt` for additional dependencies.

### Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/PaddleOCR_PyTorch_Conversion.git
    cd PaddleOCR_PyTorch_Conversion
    ```

2. **Install Dependencies**

    ```
    pip install -r requirements.txt
    ```

### Prepare Dataset

1. Update the `config.json` with the paths to your training and validation datasets.
2. Format the dataset as required by PaddleOCR, with annotations in text files as specified in `config.json`.

## Configurations

This project uses JSON files for all configurations:

- **Training & Evaluation Config**: `config.json`  
    Defines datasets, augmentations, and loaders for both training and evaluation.

- **Model Config**: `model_config.json`  
    Specifies backbone and head architectures, channel sizes, and output formats for the model.

## Usage

### Training the Model

To train the model with the specified configurations:

```
python train.py --config path/to/config.json --model_config path/to/model_config.json
```
## Evaluating the Model

To run evaluations on a validation set:

```
python eval.py --config path/to/config.json --model_config path/to/model_config.json
```
## Fine-tuning
For fine-tuning on specific layers, modify the freeze_backbone option in model_config.json.

## File Structure
```
/
├── configs/
│   ├── config.json                # Training and evaluation config
│   └── model_config.json          # Model configuration (backbone, head)
├── train.py                       # Training script
├── eval.py                        # Evaluation script
├── README.md                      # Project documentation
└── requirements.txt               # Project dependencies
```
## Contributing
Contributions are welcome! If you find any issues or want to add features, feel free to open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

## Acknowledgments
PaddleOCR - for their original OCR models and datasets.

PyTorch - for providing an efficient framework for model conversion and training.
