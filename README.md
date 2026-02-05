# Synthetic Degradation for High-Resolution Images

A Python framework for applying state-of-the-art synthetic image degradation techniques to high-resolution images. This repository focuses on noise transfer using Fast Fourier Transform (FFT) and various perturbation methods to simulate realistic image degradation patterns.

## Overview

This framework enables researchers and practitioners to:
- Extract noise characteristics from degraded images using FFT analysis
- Transfer noise patterns from one dataset to another
- Apply synthetic degradations to high-quality images for data augmentation
- Create realistic training data for image restoration and denoising tasks

The noise transfer technique uses Fourier domain analysis to isolate and extract noise characteristics, then applies them to target images while preserving their structural content.

## Features

- **FFT-Based Noise Transfer**: Extract noise patterns from degraded images and transfer them to high-quality images
- **Multiple Noise Types**: Support for Gaussian, Poisson, and Speckle noise
- **Configurable Degradation**: Adjustable noise intensity and filtering thresholds
- **Batch Processing**: Process entire datasets efficiently with progress tracking
- **Google Cloud Platform Integration**: Optimized for GCS and BigQuery workflows
- **Flexible Architecture**: Can be adapted to other Python environments beyond GCP

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/sustaz/Synthetic-Degradation-for-high-resoluted-images.git
cd Synthetic-Degradation-for-high-resoluted-images
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- Pillow - Image processing
- numpy - Numerical computations
- opencv-python - Computer vision operations
- scikit-image - Image processing algorithms
- scipy - Scientific computing and FFT operations
- pandas - Data manipulation (for GCP integration)
- tqdm - Progress bars

## Project Structure

```
.
├── src/
│   ├── fourier_noise_transfer.py      # Core noise transfer functions
│   └── fourier_perturbation_script.py # Dataset processing pipeline
├── notebooks/
│   ├── debug.ipynb                     # Debugging and testing
│   └── random_noise_trials.ipynb      # Noise experimentation
├── requirements.txt                    # Project dependencies
└── README.md                          # This file
```

## Usage

### Quick Start with Core Functions

```python
from src.fourier_noise_transfer import noise_transfer_single_img, get_noise_fourier
from scipy import fftpack
import cv2

# Load your images
img_with_noise = cv2.imread('degraded_image.jpg', cv2.IMREAD_GRAYSCALE)
img_to_degrade = cv2.imread('clean_image.jpg', cv2.IMREAD_GRAYSCALE)

# Transfer noise from one image to another
degraded_result = noise_transfer_single_img(img_with_noise, img_to_degrade)

# Save the result
cv2.imwrite('output_degraded.jpg', degraded_result)
```

### Batch Processing with GCP Integration

For processing entire datasets with Google Cloud Storage and BigQuery:

```bash
python src/fourier_perturbation_script.py \
  --gcs_source_folder gs://your-bucket/noise-source-images/ \
  --gcs_destination_folder gs://your-bucket/degraded-output/ \
  --query "SELECT * FROM your_project.dataset.table" \
  --output_counter 1000 \
  --treshold 0.03 \
  --alpha 0.5
```

### Command-Line Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `--gcs_source_folder` | string | Yes | - | GCS folder containing source images with noise patterns |
| `--gcs_destination_folder` | string | Yes | - | GCS folder for degraded output images |
| `--query` | string | No | Default query | BigQuery SQL to retrieve images for degradation |
| `--output_counter` | int | Yes | - | Number of images to process |
| `--treshold` | float | No | 0.03 | Threshold for noise filtering in Fourier domain (adjust based on noise characteristics) |
| `--alpha` | float | No | 0.5 | Weight of original signal vs. noise (0.0 = pure noise, 1.0 = pure signal) |

### Core Functions

#### `get_noise_fourier(img_fft, threshold=0.1)`
Extracts noise from an image in the Fourier domain by filtering out low-frequency components.

**Parameters:**
- `img_fft`: FFT of the input image
- `threshold`: Filtering threshold (default: 0.1)

**Returns:** Filtered FFT containing primarily noise components

#### `noise_transfer_single_img(img_perturbed, img_to_perturb, external_noise=False)`
Transfers noise from one image to another using FFT-based analysis.

**Parameters:**
- `img_perturbed`: Source image with noise
- `img_to_perturb`: Target image to degrade
- `external_noise`: Apply additional synthetic noise (default: False)

**Returns:** Degraded image with transferred noise

#### `noisy(noise_typ, image)`
Applies synthetic noise to an image.

**Parameters:**
- `noise_typ`: Type of noise ('gauss', 'poisson', 'speckle')
- `image`: Input image

**Returns:** Noisy image

## Use Cases

This framework is particularly useful for:

1. **Data Augmentation**: Generate realistic degraded training data for image restoration models
2. **Denoising Research**: Create controlled test datasets with known noise characteristics
3. **Image Quality Assessment**: Simulate various degradation patterns for quality metrics evaluation
4. **Domain Adaptation**: Prepare clean images to match the degradation patterns of a target domain
5. **Robustness Testing**: Test computer vision models against various image quality degradations

## Technical Details

### Noise Transfer Algorithm

1. **Noise Extraction**: Apply FFT to the degraded source image and filter out low-frequency components to isolate noise
2. **Noise Interpolation**: Resize the extracted noise pattern to match target image dimensions
3. **Noise Application**: Blend the noise with the target image using configurable alpha blending
4. **Optional Enhancement**: Apply additional synthetic noise types (Gaussian, Poisson, or Speckle)

### Threshold Selection

The `threshold` parameter controls how much of the frequency spectrum is considered "noise":
- Lower values (0.01-0.05): Preserve more frequency content, suitable for subtle noise patterns
- Higher values (0.1-0.3): More aggressive filtering, suitable for strong degradations

### Alpha Blending

The `alpha` parameter controls the balance between original image and noise:
- `alpha = 1.0`: Original image only (no degradation)
- `alpha = 0.5`: Equal mix of signal and noise
- `alpha = 0.0`: Pure noise pattern

## Environment Compatibility

While optimized for Google Cloud Platform, this framework can be adapted to other environments:

### Using with Local Files (Non-GCP)

Modify the I/O functions in `fourier_perturbation_script.py`:

```python
import cv2

def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

def write_image(filepath, img):
    cv2.imwrite(filepath, img)
```

### Using with AWS S3 or Azure Blob Storage

Replace `gcsfs` with appropriate cloud storage libraries (e.g., `boto3` for AWS, `azure-storage-blob` for Azure).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{synthetic_degradation,
  author = {Sustaz},
  title = {Synthetic Degradation for High-Resolution Images},
  year = {2024},
  url = {https://github.com/sustaz/Synthetic-Degradation-for-high-resoluted-images}
}
```

## License

This project is available for use under standard open-source practices. Please check the repository for specific license details.

## Acknowledgments

This framework implements noise transfer techniques based on Fourier analysis principles commonly used in image processing research.
