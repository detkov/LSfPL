# LSoPL - Label Smoothing of Pseudo Labeling
Данный репозиторий содержит код, использованный для написания бакалаврской выпускной квалификационной работы Деткова Н.С..

Тема: **Влияние label smoothing'а на псевдо-размеченные данные при обучении свёрточных нейросетей**  


Этапы для восстановления решения:  
1. Скачивание данных с [сайта](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) и распаковка в папку `input/`
2. (Опционально) Первоначальное исследование данных
    ```bash
    juputer lab
    ```
    Далее, необходимо открыть файл `src/EDA.ipynb` - в нём содержится обзор и визуализация представленных данных
3. Предобработка данных, которая формирует изображения в уменьшенном разрешении и разбиение данных для обучения на фолды
    ```bash
    cd src
    bash preprocess.sh
    cd ..
    ```
4. Запуск обучения
    ```bash
    python train.py -c exp_train_02.yaml
    ```
5. Изучение статистики по предсказаниям test сета
    ```bash
    python show_prediction_stats.py -f exp_train_02.csv
    ```
6. Генерация псевдо размеченных наборов данных для fine-tuning'а
    ```bash
    ipython Create_Datasets.ipynb
    ```
7. Генерация псевдо размеченных наборов данных с label smoothing'ом, каждый из которых является одним "экспериментом"
    ```bash
    ipython Create_Experiments.ipynb
    ```
8. Запуск экспериментов fine-tuning'а на псевдо размеченных наборах данных с label smoothing'ом
    ```bash
    ipython Run_Experiments.ipynb
    ```
9. Обзор результатов проведённых экспериментов
    ```bash
    juputer lab
    ```
    Далее, необходимо открыть файл `src/Analyse_Experiments_Results.ipynb` - в нём содержится обзор и визуализация представленных данных

Сам диплом расположен в файле `thesis.pdf`.