# 07.06.2020
Начал работу. 

* Скачал файлы `.csv` и сам датасет (32Гб) из фотографий ;
* Сделал скрипт, который многопоточно ресайзит все фотографии в нужное разрешение;
* Сделал скрипт, разбивающий train на фолды, при этом, есть возможность выделять часть выборки как валидационную;
* Сделал класс `MelanomaDataset`, а также нашел корректные аугментации;
* Сделал модель, переделал `MelanomaDataset` и трансформации;

# 08.06.2020
* Первые 5 моделей обучились, процесс обучения лежит в `exp_01.txt`;
* Не смог сделать нормальный инференс, поэтому переобучаю на другом кернеле;
* Запустил обучение на других параметрах;

# 09.06.2020
* Понял, что надо уменьшить `LR`, сделать GroupKFold и добавить TTA;
* Сделал разбитие по GroupKFold (`train_folds_2.csv`);
* Уменьшил `LR`, заменил `AdamW` на `Adam` - результат много хуже, заменяю обратно;
* Ресайзнул изображения до `256x256`;
* Добавил `d4_transform` как TTA (`hflip + rotation [0, 90, 180, 270]`);
* Добавил считывание конфига;
* Добавил отдельный скрипт для обучения;
* Поправил код, исходя из замечаний ревьюера;
* Понял, что надо попробовать `CyclicLR`;
* Третий эксперимент оказался ужасным по скору;
* Осознал, что нужно также разбивать и валидационную выборку групповым методом, исправил код;
* Добавил в код вычисление ROC AUC по hold-out сету;
* Изменил TTA на `hflip + vflip + rotation [0] (исходное)`;
* Сейчас сделаю последнюю попытку с `256x256`, `ReduceLROnPlateau`, изменённой аугментацией, TTA и групповыми фолдами;

# 10.06.2020
* Код упал с ошибкой на вычислении результата на hold-out сете;
* Запустил все эксперименты на hold-out сетах;
* Снова замени TTA на `d4_transform`;
* Запустил пятый эксперимент;
* Исследовал таргет, предстазанный для теста: понял, что эксперименты с разрешением `256x256` предсказывают ничтожно мало сэмплов с классом 1: 6 из 10982 - это 0.05%, хотя количество в тесте примерно 1.76%. Посмотрел на второй эксперимент (`512x512`), там уже 0.52% - уже хоть что-то. Далее написал какие эксперименты с этим таргетом хочется провести и как их можно улучшить. См. `Experiments.md`;
* Понял, что надо было использовать FocalLoss;
* Сделал скрипт, выводящий статистику и распределения по предсказанному таргету;
* Пятый эксперимент прошел неуспешно, с классом 1 только 5 сэмплов;
* Начал 6 эксперимент, калька со второго с наилучшим результатом;

# 11.06.2020
* 6 эксперимент показал очень высокий ROC AUC(0.914) на kaggle, на hold-out выборке, но дал всего 9 предсказаний класса 1;
* Запустил 7 эксперимент со стратифицированными фолдами и большей `ES_PATIENCE`;
* Задумался о том, что. возможно, эти модели, которые дают хорошие скоры, просто идельно предсказывают все классы 1, но а остальные, в которых они уверены не настолько, просто относятся к классу 0. Можно попробовать брать для псевдо-лэйблинга вообще все вероятности, начиная от 1, пока не наберётся нужный процент от всей выборки;
* Создал скрипт, который создаёт `.csv` со всеми экспериментами;
* Не получается усреднить веса из-за `RuntimeError: Error(s) in loading state_dict for EfficientNet: ...`;
* Сделал возможность учить только последний слой;
* Сделал скрипт для файн-тюнинга сети. Ожижается, что один 5-эпоховый эксперимент с обучением только последнего слоя будет занимать ~17 минут, вычисление метрики на hold-out сете будет занимать дополнительные ~17 минут, и как результат - 34 минуты на 1 эксперимент.  
Если прикинуть, то при обучении не только последнего слоя, одна эпоха будет длиться ~10 минут, что резко поднимает время выполнения одного эксперимента.  
Следовательно, думаю, проведу лишь эксперименты с обучением головы сети;
* Провел эксперимент и сравнил результаты файн-тюнинга с обучением на 3 и 5 эпохах (`exp_train_02`) на hold-out сете:
    * `Исходное: 0.8263`
    * `3 epochs: 0.8550`
    * `5 epochs: 0.8542`
  
  Следовательно, оставляю 3 эпохи. В таком случае время 1 эксперимента сокращается до 27 минут при, *скорее всего*, лучшем результате;

* Сделал скрипт с прогонкой всех экспериментов, запустил его.  
Ожидаю, что за 16 часов должен завершиться;

# 12.06.2020
* За 17 часов посчитались все эксперименты;
* Написал больше половины диплома и почти весь отчет по практике;

# 13.06.2020
* Понял, что необходимо нагенерировать кучу графиков, чтобы забивать как-то место, начинаю этим заниматься;
* Сделал визуализацию для EDA;
* Дописал всё до экспериментов с pseudo-labeling, fine-tuning и label smoothing'ом;

# 14.06.2020
* Вторая партия экспериментов с обучением всей сети закончилась за 35 часов, предварительный вывод таков, что такой тип обучения принёс результата хуже;
* Обновил скрипт, показывающий статистику по предсказаниям, так как понял, что `round(0.5)`  - это 0, а не 1;
* Пришла идея для ещё одного класса экспериментов: учить не только на псевдо-лэйблах, но на трейне + псевдо-лэйблах, но в таком случае надо делать балансировку лосса, что тоже добавляет сложностей;
* Дописал диплом;

# 15.06.2020
Закончил работу.