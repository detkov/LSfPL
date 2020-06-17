# LSoPL - Label Smoothing of Pseudo Labeling
This repository contains the code used to accomplish N.S. Detkov's bachelor's thesis.

Bachelor's Thesis: **Influence of the label smoothing of pseudo labeled data on training CNN**


Steps to reconstruct the solution:  
1. Download the data from the [site](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) and unpack in the `input/` folder
2. (Optional) Exploratory Data Analysis
    ```bash
    juputer lab
    ```
    Further, you need to open the `src/EDA.ipynb` file - it contains an overview and visualization of the presented data
3. Data pre-processing, which generates reduced resolution images and splits the training data into folds
    ```bash
    cd src
    bash preprocess.sh
    cd ..
    ```
4. Launch of the training process
    ```bash
    python train.py -c exp_train_02.yaml
    ```
5. Review the statistics on the predictions of the test set
    ```bash
    python show_prediction_stats.py -f exp_train_02.csv
    ```
6. Generation of pseudo labeled datasets for the fine-tuning
    ```bash
    ipython Create_Datasets.ipynb
    ```
7. Generation of pseudo labeled datasets applying label smoothing, each of which is "experiment"
    ```bash
    ipython Create_Experiments.ipynb
    ```
8. Launch fine-tuning experiments on pseudo label with label smoothing
    ```bash
    ipython Run_Experiments.ipynb
    ```
9. Overview of the experiments results
    ```bash
    juputer lab
    ```
    Further, you need to open the `src/Analyse_Experiments_Results.ipynb` file - it contains an overview and visualization of the presented data

The thesis itself can be found in the `thesis.pdf` file.