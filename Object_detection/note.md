
# 需要整個再 review 一遍

### 使用 PN Learning
    包含 Positive train dataset 跟 Negative train dataset

0. dataset 包含了 image file 跟 annotation of object information in image

1. 從 dataset 中隨機取出一部份來當作 positive training data
    extract_features.py

2. 需選擇一組 dataset 來當 negative training data, 例如若要偵測汽車則可以選擇風景照來當 negative training data ***要偵測的物件必須不能出現在 negative training image 裡***
    extract_features.py

3. 接著對 Positive training data 跟 Negative training data 以 HOG Descriptor 擷取 featrues， 
    1. Positive training data 的 feature 標為 1
    2. Negative training data 的 feature 標為 －1
    extract_features.py

4. 將步驟 3 的 features 存到檔案(存成 HDF5 的檔案)
    extract_fetures.py
5. 使用步驟 4 的 features file 來訓練 Linear SVM model 並將 model 存到檔案(cPickle)
    train_model.py
6. 最後輸入照片測試結果
    test_mode_no_nms.py

