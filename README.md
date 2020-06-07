# LSfPL
Bachelor Thesis: Influence of the label smoothing for pseudo labeled data for training CNN  

Тема диплома: Влияние label smoothing'а на псевдо-размеченные данные при обучении свёрточных нейросетей

# TODO

- [ ] EDA
- [ ] Implement first solution:
  - [x] Resize Dataset
  - [ ] Create folds split
  - [ ] Create Dataset module 
  - [ ] Find relevant augmentations
  - [ ] Create Loss function 
  - [ ] Create Model
  - [ ] Train, predict and make first submit
- [ ] Achieve best possible score:
  - [ ] Try different augmentations
  - [ ] Try different models 
  - [ ] Try different optimizers (SWA, RAdam, LARS, AdamW, Ralamb, LookAHead, etc.)
  - [ ] Try combining weights of different epochs 
  - [ ] Make OOF prediction 
  - [ ] Make ensemble of models
  - [ ] Try label post-processing 
- [ ] Research influence of the label smoothing for pseudo labeling
  - [ ] ...