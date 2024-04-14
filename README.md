# SETransformerFusion: A Multimodal Fusion Model for Image and Blood Data

SETransformerFusion is a novel multimodal fusion model that combines visual features from images and blood test data to perform classification tasks. It leverages the power of Squeeze-and-Excitation (SE) blocks and Transformer architecture to effectively capture and fuse the information from both modalities.

## Model Architecture

The SETransformerFusion model consists of the following components:

1. **Image Embedding**: A linear layer that projects the image features into a hidden dimension space.
2. **Blood Embedding**: A linear layer that projects the blood test features into the same hidden dimension space.
3. **SE Blocks**: Squeeze-and-Excitation blocks are applied to both the image and blood embeddings to adaptively recalibrate the channel-wise feature responses.
4. **Positional Encoding**: Positional encoding is added to the fused image and blood embeddings to incorporate positional information.
5. **Transformer Encoder**: A Transformer encoder with multiple layers and multi-head attention is used to capture the interactions and dependencies between the image and blood features.
6. **Output Layer**: A fully connected layer maps the fused features to the desired number of classes.

## Features

- Multimodal fusion of image and blood test data
- Utilizes Squeeze-and-Excitation blocks to enhance feature representation
- Employs Transformer architecture to capture complex interactions between modalities
- Supports continual learning to adapt to new classes with limited samples

## Usage

1. Prepare your image dataset and blood test data in the required format.
2. Initialize the SETransformerFusion model with the desired hyperparameters.
3. Train the model using the provided training loop and evaluate its performance.
4. If a new class is encountered with low confidence, the model can perform continual learning to adapt to the new class.

## Results

The SETransformerFusion model has shown promising results in combining visual and blood test information for classification tasks. It achieves high accuracy on both the training and testing datasets, demonstrating its ability to effectively fuse multimodal data.

## Contributions

Contributions to the SETransformerFusion model are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
