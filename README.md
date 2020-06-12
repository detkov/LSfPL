# LSfPL
Bachelor Thesis: Influence of the label smoothing for pseudo labeled data for training CNN  

Тема диплома: Влияние label smoothing'а на псевдо-размеченные данные при обучении свёрточных нейросетей

# TODO

- [x] EDA
- [x] Implement first solution:
  - [x] Resize Dataset
  - [x] Create folds split
  - [x] Create Dataset module 
  - [x] Find relevant augmentations
  - [x] Create Loss function 
  - [x] Create Model
  - [x] Start Training
  - [x] Train, predict and make first submit
- [ ] Achieve best possible score:
  - [ ] Try different augmentations
  - [x] Try different models 
  - [ ] Try different optimizers (SWA, RAdam, LARS, AdamW, Ralamb, LookAHead, etc.)
  - [ ] Try combining weights of different epochs 
  - [x] Make OOF prediction 
  - [ ] Make ensemble of models
  - [ ] Try label post-processing 
- [ ] Research influence of the label smoothing for pseudo labeling
  - [ ] ...