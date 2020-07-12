# Бакалаврская выпускная квалификационная работа
> Thesis can be found here: [`bs_thesis_en.pdf`](https://github.com/detkov/LSoPLD/blob/master/bs_thesis_en.pdf)  
> Диплом расположен тут: [`bs_thesis_ru.pdf`](https://github.com/detkov/LSoPLD/blob/master/bs_thesis_ru.pdf)

Данный репозиторий содержит код, использованный для написания бакалаврской выпускной квалификационной работы Деткова Н.С..

Тема: *Влияние label smoothing'а псевдо-размеченных данных на обучение свёрточных нейронных сетей*

Этапы для восстановления решения:  
1. Скачивание данных с [сайта](https://www.kaggle.com/c/siim-isic-melanoma-classification/data) и распаковка в папку `input/`
2. (Опционально) Первоначальное исследование данных
    ```bash
    cd src
    juputer lab
    ```
    Далее, необходимо открыть файл `EDA.ipynb` — в нём содержится обзор и визуализация представленных данных
3. Предобработка данных, которая формирует изображения в уменьшенном разрешении и разбиение данных для обучения на фолды
    ```bash
    bash preprocess.sh
    ```
4. Запуск обучения
    ```bash
    python train_infer.py -c exp_train_02.yaml
    ```
5. Изучение статистики по предсказаниям test сета
    ```bash
    python show_prediction_stats.py -f exp_train_02.csv
    ```
6. Генерация псевдо-размеченных наборов данных для fine-tuning'а
    ```bash
    ipython Create_Datasets.ipynb
    ```
7. Генерация псевдо-размеченных наборов данных с label smoothing'ом, каждый из которых является одним "экспериментом"
    ```bash
    ipython Create_Experiments.ipynb
    ```
8. Запуск экспериментов fine-tuning'а на псевдо-размеченных наборах данных с label smoothing'ом
    ```bash
    ipython Run_Experiments.ipynb
    ```
9. Обзор результатов проведённых экспериментов
    ```bash
    juputer lab
    ```
    Далее, необходимо открыть файл `Analyse_Experiments_Results.ipynb` — в нём содержится обзор и визуализация представленных данных