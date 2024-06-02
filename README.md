# Semantic Segmentation on Cityscapes Dataset

## Project Overview

The goal of this project is to train a model for the semantic segmentation task on the Cityscapes dataset, utilizing cross-validation and evaluating performance using the Mean Intersection over Union (MIoU) metric.

## Data Preparation

To prepare the data for training, `gtFine_trainval` was used for annotations and `leftImg8bit_trainval` for the images. The provided reference Cityscapes scripts were utilized to prepare the training labels using the `csCreateTrainIdLabelImgs` script.

## Model Architecture

The U-Net architecture with a ResNet-34 encoder pretrained on ImageNet was chosen due to its proven effectiveness and efficiency in segmentation tasks.

## Training Details

- **Cross-Validation**: A 5-fold cross-validation was utilized for a robust evaluation on the training dataset. The final model version was trained on the complete training data and evaluated using the provided validation dataset.
- **Hyperparameters**: Batch size of 4, learning rate of 1e-4.
- **Optimizer**: Adam optimizer.
- **Loss Function**: CrossEntropyLoss (ignoring the 255 index for unlabeled pixels).
- **Training Strategy**: The model was trained for 10 epochs for each fold, saving the best model based on validation loss.

## Results

- **Average MIoU across 19 classes**: 66.7
- **MIoU on test dataset**: 66.7901

## Discussion

The model achieved an average MIoU of 66.7901, which is a fairly decent result compared to the benchmark models on the Cityscapes website. This performance indicates that the U-Net architecture with a ResNet-34 encoder is effective for semantic segmentation tasks on the Cityscapes dataset. Some challenges encountered included using the Cityscapes script package, which was not very robust and required troubleshooting to prepare the training labels correctly. Additionally, I experimented with various data augmentations such as `RandomHorizontalFlip()`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`, and Gaussian blur during data preparation before preprocessing for the pretrained model. However, these augmentations did not result in improved performance. Despite these challenges, the cross-validation approach provided a reliable evaluation of the model.

## Running the Code

To run the code, follow these steps:

1. **Download the Data**: First, download the Cityscapes dataset and save it in a folder named `cityscapes` in the root directory.
2. **Train the Model**:

   - Open the training notebook and run the cells sequentially.
   - This notebook fine-tunes the U-Net model with a ResNet-34 encoder on the Cityscapes dataset.
   - Ensure all necessary packages are installed, including `segmentation_models_pytorch`, `torch`, `numpy`, `PIL`, and others used in the notebook.
3. **Evaluate the Model**:

   - Once training is complete, open the evaluation notebook and run the cells sequentially.
   - This notebook performs inference to generate predicted segmentation maps and uses the `csEvalPixelLevelSemanticLabeling` script to compute the Mean Intersection over Union (MIoU) score.

The `src` directory also includes two folders, `predictions` and `test_predictions`. These folders contain the images generated from inference on the validation and test data, respectively, and were used to calculate the MIoU score.

## Sources

- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

Training curves and results for the cross-validation folds are included in the report.
