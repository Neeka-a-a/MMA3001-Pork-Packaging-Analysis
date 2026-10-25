
# MMA3001 – Pork Packaging Defect Analysis

## Project Overview

This project investigates the use of numerical image processing and regression methods to identify unsealed pork packaging.

The project was developed for **MMA3001: Numerical Methods and Machine Learning** at Monash University.

Two regression methods were investigated:
- Linear Regression
- Decision Tree Regression

The models use numerical image features rather than raw image pixels to predict whether a packaging image contains an **unsealed** defect annotation.

## Dataset

**Source:** Pork Rasher Error (Packaging), Version 4  
**Provider:** Roboflow Universe  
**Dataset:** https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4  
**Annotation format:** COCO JSON

The original dataset contains **3,549 images**.

| Dataset | Original images | Corrected images |
|---|---:|---:|
| Training | 3,349 | 2,667 |
| Validation | 120 | 369 |
| Testing | 80 | 418 |
| Total | 3,549 | 3,454 |

### Target Variable

The project focuses on the `unsealed` annotation category.

- **0:** No unsealed annotation recorded.
- **1:** At least one unsealed annotation recorded.

A label of 0 does not necessarily confirm that the package is correctly sealed.

## Image Feature Extraction

Two feature extraction methods were investigated.

### Whole-Image Features

Four measurements are calculated across the entire image:

1. Mean greyscale intensity
2. Greyscale standard deviation
3. Mean HSV saturation
4. Canny edge density

This produces **4 numerical features per image**.

### Regional Features

Each image is resized to 300 × 300 pixels and divided into a 3 × 3 grid.

The same four measurements are extracted from each of the nine regions.

**9 regions × 4 measurements = 36 features per image**

Regional features were selected for the final regression experiments because they retain more spatial image information.

Feature extraction is implemented in:

`src/preprocessing.py`

## Data Leakage Prevention

The original dataset contained overlapping and closely numbered production-line frames across its training, validation and testing partitions.

To reduce potential data leakage:

1. Frame identifiers were extracted from image filenames.
2. Images were grouped into blocks of 20 frame identifiers.
3. `GroupShuffleSplit` assigned complete frame blocks to dataset partitions using `random_state=42`.
4. Training images within five frame numbers of validation or testing images were excluded.
5. Final checks confirmed that no cross-partition frame identifiers differed by five or fewer numbers.

This removed **95 training images**.

The corrected dataset contains **3,454 images**.

The final assignments are stored in:

`data/final_grouped_manifest.csv`

Images remain in their original Roboflow folders. The manifest records their corrected partition assignments.

## Computational Methods

### Linear Regression

Linear regression predicts a continuous numerical score using a weighted combination of the 36 input features.

The score is converted into a binary classification using a threshold.

### Decision Tree Regression

Decision tree regression divides the numerical feature space through successive decision rules.

Different maximum tree depths and classification thresholds were investigated.

### Baseline Models

Two simple baselines were also evaluated:

- **Always Unsealed:** Predicts every image as unsealed.
- **Never Unsealed:** Predicts every image as not unsealed.

These baselines provide reference performance levels for comparison with the regression models.

## Model Optimisation

Classification thresholds between 0.10 and 0.80 were investigated in increments of 0.05.

Decision tree maximum depths of 2, 3, 4, 5, 6, 8 and 10 were evaluated.

The selected configurations were:

| Parameter | Linear Regression | Decision Tree |
|---|---|---|
| Input features | 36 | 36 |
| Classification threshold | 0.30 | 0.30 |
| Maximum tree depth | N/A | 8 |

Model settings were selected using validation performance, with F1-score as the primary selection metric.

The test partition was reserved for final evaluation.

## Model Performance

### Corrected Validation Dataset

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Always Unsealed | 0.409 | 0.409 | 1.000 | 0.581 |
| Never Unsealed | 0.591 | 0.000 | 0.000 | 0.000 |
| Linear Regression | 0.840 | 0.764 | 0.881 | 0.818 |
| Decision Tree Regression | 0.748 | 0.730 | 0.609 | 0.664 |

### Final Held-Out Test Results

The selected linear regression model was evaluated on 418 previously withheld test images.

| Metric | Result |
|---|---:|
| Accuracy | 0.7488 |
| Precision | 0.6482 |
| Recall | 0.7866 |
| F1-score | 0.7107 |
| False positives | 70 |
| False negatives | 35 |

The results demonstrate the potential of numerical feature-based packaging classification but do not establish readiness for industrial deployment.

## Computational Runtime

Model training and prediction times were measured using `time.perf_counter()` over ten repetitions.

| Measurement | Linear Regression | Decision Tree |
|---|---:|---:|
| Median training time | 8.59 ms | 238.39 ms |
| Prediction time for 369 images | 0.368 ms | 0.657 ms |
| Prediction time per image | 0.0010 ms | 0.0018 ms |

These measurements exclude image loading and feature extraction.

## Project Structure

The main project files are organised as follows:

```text
MMA3001-Pork-Packaging-Analysis/
│
├── data/
│   ├── raw/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── final_grouped_manifest.csv
│
├── notebooks/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── regression.py
│   └── evaluation.py
│
├── tests/
│   └── test_preprocessing.py
│
├── results/
│
├── docs/
│
├── requirements.txt
└── README.md
```

## Installation and Reproducibility

### Requirements

- Python 3
- NumPy
- pandas
- Matplotlib
- OpenCV
- Pillow
- scikit-learn
- pytest

Additional project dependencies are listed in `requirements.txt`.

### Download the Project

```bash
git clone https://github.com/Neeka-a-a/MMA3001-Pork-Packaging-Analysis.git

cd MMA3001-Pork-Packaging-Analysis
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the Notebook

Open the project notebook from the `notebooks/` directory using Jupyter Notebook or Google Colab.

Run the notebook cells in order.

The notebook:

1. Locates the project and verifies the original dataset.
2. Loads COCO annotations.
3. Examines defect distributions and image examples.
4. Extracts whole-image and regional numerical features.
5. Investigates potential sequence-related data leakage.
6. Constructs corrected dataset partitions.
7. Trains regression models.
8. Optimises classification thresholds and tree depth.
9. Evaluates the models against baseline methods.
10. Performs final held-out test evaluation.
11. Exports performance results.

**Important:** Reproduction requires the original image files and COCO annotation files to be available under `data/raw/`, along with the corrected dataset manifest.

The notebook uses portable project paths and does not require Google Drive access or personal authentication once the required files are available locally.

## Automated Testing

Run the preprocessing tests from the project root:

```bash
python -m pytest tests/test_preprocessing.py -v
```

The tests check image feature extraction, missing-image handling, finite numerical outputs and regional feature dimensions.

## Results

The notebook generates output files in the `results/` directory, including:

- `model_comparison.csv`
- `threshold_optimisation.csv`
- `tree_optimisation.csv`
- `runtime_comparison.csv`

Graphs generated by the notebook illustrate threshold optimisation, model performance and computational runtime.

## Documentation

The Python functions contain docstrings describing their purpose, parameters, return values and relevant error handling.

HTML documentation can be generated using `pdoc`.

## Limitations

- The models use numerical image features rather than direct seal-integrity measurements.
- Existing COCO annotations are treated as reference labels and may contain errors.
- A negative label does not independently verify that a package is correctly sealed.
- Frame-based separation reduces potential leakage but does not guarantee physical independence between packages.
- Model runtime measurements exclude image preprocessing.
- The selected model still produces false positives and false negatives.

Further testing would be required before use in an industrial quality-control environment.

## AI Use

ChatGPT (OpenAI) was used to assist with understanding computational methods, developing and debugging code, generating docstrings and comments, interpreting results and improving report writing.

AI-assisted outputs were checked through code execution, data inspection and automated testing. Final engineering decisions, results and conclusions remain the author's responsibility.

## References

**Roboflow Dataset:**  
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging/dataset/4

**OpenCV:**  
https://opencv.org/

**NumPy:**  
https://numpy.org/

**Scikit-learn:**  
https://scikit-learn.org/

## Licence and Dataset Attribution

The Pork Rasher Error (Packaging) dataset was obtained from Roboflow Universe and is attributed to its original publisher.

Dataset licensing and attribution requirements should be followed when redistributing the images.

The project was developed for educational purposes as part of MMA3001.
