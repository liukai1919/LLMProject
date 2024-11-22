# LLM Project

## Project Task
- **Objective**: Perform sentiment analysis on IMDb reviews to classify them as positive or negative.
- **Purpose**: Evaluate and improve sentiment classification models for better performance.

## Dataset
- Preprocessed IMDb review dataset (ready for use).
- Manually labeled dataset and tokenized.
- Transform them into a DataFrame.
- Use RandomForest to train a model.

## Pre-trained Model
- Download a pre-trained model from Hugging Face (`distilbert/distilbert-base-uncased`).
- Build a pipeline for sentiment analysis.
- Use the Trainer API to fine-tune the model.
- Calculate the accuracy of the model.

## Performance Metrics
- **Evaluation Loss**: 0.2059
- **Evaluation Accuracy**: 92.08%
- **Evaluation Runtime**: 473.18 seconds
- **Evaluation Samples per Second**: 52.834
- **Evaluation Steps per Second**: 3.303
- **Epoch**: 2.0

## Hyperparameters
1. **output_dir**: Specifies the directory where the model and checkpoints will be saved. Ensure you have enough storage space and that the path is valid.
2. **learning_rate**: The learning rate controls the step size during model updates. A smaller learning rate often results in more stable training but may require more time to converge.
3. **per_device_train_batch_size** and **per_device_eval_batch_size**: The batch size for training and evaluation per device. The batch size affects memory usage and training speed.
4. **num_train_epochs**: The number of training epochs. More epochs can improve model performance but may also lead to overfitting.
5. **weight_decay**: Weight decay is used for regularization to prevent overfitting.
6. **eval_strategy** and **save_strategy**: Evaluation and saving strategies, set to `"epoch"` means evaluation and saving occur at the end of each training epoch.
7. **load_best_model_at_end**: If set to `True`, the best model is loaded at the end of training.
8. **push_to_hub**: If set to `True`, the model is pushed to the Hugging Face Hub after training.

These parameters directly impact the training process and the final performance of the model, so they should be adjusted based on the specific task and available resources.