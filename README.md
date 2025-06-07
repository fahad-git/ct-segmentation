# ModSegNet3D: A Modular Non-Contrast CT Segmentation Pipeline Integrating 3D Attention Mechanisms and Large Language Models

This repository contains the code and resources for the Master's thesis "ModSegNet3D: A Modular Non-Contrast CT Segmentation Pipeline Integrating 3D Attention Mechanisms and Large Language Models," completed at the Norwegian University of Science and Technology (NTNU).

## Overview

Accurate segmentation of cardiac structures from Computed Tomography (CT) is crucial for the diagnosis and management of Cardiovascular Diseases (CVDs). While Contrast-Enhanced CT (CECT) offers clear images, it involves risks and costs associated with contrast agents. Non-Contrast CT (NCCT) is a safer alternative but presents challenges for automated segmentation due to low tissue contrast and limited annotated datasets.

ModSegNet3D is a modular deep learning pipeline designed to address these challenges in NCCT cardiac segmentation. It integrates comprehensive preprocessing, a 3D Convolutional Block Attention Module (CBAM) within the nnU-Net framework.

## Features

* **Modular Pipeline:** Encompasses data conversion, preprocessing, model training, and post-processing.
* **3D Attention Mechanism:** Integrates a 3D CBAM into the nnU-Net architecture to improve feature representation and segmentation accuracy in low-contrast NCCT images.
* **NCCT Focus:** Specifically designed and optimized for segmenting the whole heart and left ventricle in NCCT scans.
* **Label Registration:** Includes a method to transfer CECT annotations to NCCT images, expanding the training data.

---
![Screenshot 2025-06-04 at 2 22 18 PM](https://github.com/user-attachments/assets/ab480cf8-a191-42e3-a36f-a52745e1647e)

---
Architecture


![Screenshot 2025-06-04 at 2 23 57 PM](https://github.com/user-attachments/assets/9655c4eb-bb9a-42e4-a1bb-cc9c784d901e)

---

## Methodology

The ModSegNet3D pipeline involves several key stages:

1.  **Data Conversion:** DICOM files are converted to NIFTI format, with attention to coordinate system differences (LPS to RAS) and voxel spacing alignment.
2.  **Data Preparation and Preprocessing:**
    * **Fingerprint Extraction:** Analyzes the dataset to automatically configure network architecture, patch size, and training parameters.  
    * **Standard Preprocessing:** Includes image resampling, intensity normalization (e.g., z-score for CT), cropping, padding, and data splitting for K-fold cross-validation. 
3.  **Proposed Model Architecture (ModSegNet3D):**
    * **Base:** nnU-Net framework.  
    * **Enhancement:** Integration of a 3D Convolutional Block Attention Module (CBAM) into the decoder part of the nnU-Net architecture. 
    * **Loss Function:** Compound loss combining Dice loss and Cross-Entropy loss.  
    * **Optimizer:** Stochastic Gradient Descent (SDG) with Nesterov momentum.  
4.  **Model Training:**
    * **Patch Extraction:** Employs a patch-based training approach.  
    * **Data Augmentation:** Uses spatial (rotations, scaling, elastic deformations, flipping) and intensity-based (Gaussian noise/blur, brightness/contrast adjustments, gamma correction) augmentations.  
    * **K-Fold Cross Validation:** Utilizes 5-fold cross-validation.  
    * **Inference Strategy:** Employs ensembling of models from each fold.  
5.  **Post-processing:** Applies 3D Connected Component Analysis to refine segmentation outputs by removing small, inaccurate predictions. 
 
---

![Screenshot 2025-06-04 at 2 44 11 PM](https://github.com/user-attachments/assets/708357bc-dfbb-4ff6-9d43-ed782afd6c5c)

---

## Results

* ModSegNet3D demonstrated strong performance in segmenting the whole heart and left ventricle on NCCT data.  
* The inclusion of 3D CBAM improved feature representation and segmentation accuracy, particularly in low-contrast conditions.  
* The pipeline outperformed baseline nnU-Net, 3D U-Net, and SegFormer3D on the NCCT dataset.  
    * Achieved a Dice score of 93.5% and IoU of 87.9% for whole heart segmentation. 
    * Achieved a Dice score of 78.6% and IoU of 65.4% for left ventricle segmentation.

---

![Screenshot 2025-06-04 at 2 28 10 PM](https://github.com/user-attachments/assets/3f044252-4a3b-4a78-9e81-e9a8b587b8c0)

---

![Screenshot 2025-06-04 at 2 28 42 PM](https://github.com/user-attachments/assets/038e7f39-9fd3-4417-87db-396c235e3f54)

---

## Usage

1.  **Prerequisites:**
    * Python 3.10
    * PyTorch 2+ (with CUDA 12 if using GPU)
    * `dcm2niix`
    * `NiBabel`
    * `NumPy`
    * For more details: [visit](https://github.com/fahad-git/ct-segmentation/blob/main/ModSegNet3D(nnU-Net)/documentation/installation_instructions.md)
2.  **Dataset Preparation:**
    * Organize your DICOM/NIFTI data as expected by the pipeline.
    * Run the data conversion scripts presented inside `data_conversion_and_preparation` directory.
    * Execute the preprocessing steps (for more details [visit](https://github.com/fahad-git/ct-segmentation/blob/main/ModSegNet3D(nnU-Net)/documentation/how_to_use_nnunet.md)).
3.  **Training:**
    * Configure the training parameters in the respective JSON files.
    * Run the training script (for more details [visit](https://github.com/fahad-git/ct-segmentation/blob/main/ModSegNet3D(nnU-Net)/documentation/how_to_use_nnunet.md)).
4.  **Inference:**
    * Use the trained models to predict segmentation on new data (for more details [visit](https://github.com/fahad-git/ct-segmentation/blob/main/ModSegNet3D(nnU-Net)/documentation/how_to_use_nnunet.md))
5.  **Evaluation:**
    * Run evaluation scripts to calculate Dice, IoU, etc.

## Code Acknowledgments

This repository builds upon and utilizes code from the following open-source projects. We are grateful to their authors and contributors:

* **nnU-Net Framework:** The core self-configuring framework for medical image segmentation.
    * [https://github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* **Dynamic Network Architectures:** Used as a base for CBAM integration.
    * [https://github.com/MIC-DKFZ/dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)

* **Attention Modules (for CBAM):** Provided the foundational attention mechanisms adapted for 3D CBAM.
    * [https://github.com/Jongchan/attention-module](https://github.com/Jongchan/attention-module)


## Acknowledgments

* Dr. Mohib Ullah (Supervisor) 
* Prof. Faouzi Alaya Cheikh (Co-supervisor) 
* Dr. Ali Shariq Imran (Co-supervisor) 
* Dr. Øyvind Nordbø (Co-supervisor) 
* Norsvin AS for providing the dataset and resources. 
* NORPART-CONNECT project, funded by Diku, for financial support.  
 
