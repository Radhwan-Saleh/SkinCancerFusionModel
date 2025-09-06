# Skin Cancer Classification - Fusion_Model

This repository contains the official TensorFlow implementation for the paper: **"Explainable Depth-Wise and Channel-Wise Fusion Models for Multi-Class Skin Lesion Classification"**

Our work presents a systematic investigation into deep feature fusion models for multi-class skin lesion classification. This repository provides the code to train our proposed architectur FM6, evaluate their performance, and generate the Explainable AI (XAI) visualizations (Grad-CAM and SHAP) as presented in the article.

---

## Repository Structure

-   `FusionModel.py`: The main script for loading data (HAM10000), defining the model architectures, training, evaluation, and generating XAI plots.
-   `GradCAM_module.py`: A utility module containing the class and functions for generating Grad-CAM heatmaps.
-   `CM_module.py`: A utility module for plotting the confusion matrix (Note: you will need to add your `plot_confusion_matrix` function to this file).
-   `requirements.txt`: A list of all Python dependencies required to run the code.
-   `LICENSE`: The MIT License file.

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Radhwan-Saleh/SkinCancerFusionModel.git
cd SkinCancerFusionModel
```

### 2. Create a Virtual Environment (Recommended)

It is highly recommended to create a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install all the required packages using the `requirements.txt` file.
(Note: we used Tensorflow 2.14)

```bash
pip install -r requirements.txt
```

### 4. Dataset Preparation

This code relies on the public **HAM10000 dataset**.

1.  **Download the dataset:** You can find it on Kaggle or other academic sources.
2.  **Structure the data:** The script expects the dataset to be organized into `train` and `val` directories, with subdirectories for each of the 7 classes. Your final data structure should look like this:

    ```
    /path/to/HAM10000/
    ├── train/
    │   ├── AKIEC/
    │   ├── BCC/
    │   ├── BKL/
    │   ├── DF/
    │   ├── MEL/
    │   ├── NV/
    │   └── VASC/
    └── val/
        ├── AKIEC/
        ├── BCC/
        ├── BKL/
        ├── DF/
        ├── MEL/
        ├── NV/
        └── VASC/
    ```

3.  **Update Paths in `FusionModel.py`:** Before running, you **must update the hardcoded paths** in the `FusionModel.py` script to point to your dataset location. Look for the variable `main_path` and other similar path definitions.

---

## How to Use the Code

The `FusionModel.py` script is organized into sections based on the comments (`In[0]`, `In[1]`, etc.). You can run these sections sequentially in an interactive environment like Spyder, VSCode, or a Jupyter Notebook.

### 1. Model Training

-   Ensure your dataset is correctly structured and paths are updated.
-   Run the sections **`In[0]`** (data loading) and **`In[1]`** (model definition).
-   Execute section **`In[2]`** to start the training process. Model weights (`.keras` files) and training history (`.csv` file) will be saved in your working directory.

### 2. Reporting Results and Evaluation

-   After training, load the weights of your best model by updating the `model_path` variable in section **`In[3.1]`**.
-   Run section **`In[4]`** and its subsections to:
    -   Generate and display the **confusion matrix** (`In[4.1]`).
    -   Print and save the **classification report** (`In[4.2]`).
    -   Compute and plot the **ROC curves** for each class (`In[4.3]`).
    -   Save predicted probabilities to an Excel file (`In[4.4]`).

### 3. Generating Explainable AI (XAI) Visualizations

#### Grad-CAM

-   Load a trained model first (section `In[3.1]`).
-   In section **`In[3.2]`**, specify the class (`skin_class`) and image paths (`img_paths`) you want to visualize.
-   Run section **`In[3.3]`** to generate and save the Grad-CAM plots to the specified output directory.

#### SHAP

-   Load a trained model first (section `In[3.1]`).
-   Run section **`In[5.1]`** to prepare the data for the SHAP explainer.
-   In section **`In[5.3]`**, specify the class (`skin_class`) and image paths (`img_paths`) for visualization.
-   Run section **`In[5.4]`** to generate and save the SHAP value plots.

---

## Citation

If you use this code or our work in your research, please cite our paper:

```bibtex
@article{AbuAlkebash2025,
  title={Explainable Depth-Wise and Channel-Wise Fusion Models for Multi-Class Skin Lesion Classification},
  author={AbuAlkebash, Humam and Saleh, Radhwan A. A. and Ertunç, H. Metin},
  journal={PLOS ONE},
  year={2025}
}
```

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Acknowledgments

This work utilizes the HAM10000 dataset, collected and made publicly available by Tschandl et al. We thank the authors and contributors for their invaluable contribution to the research community.
