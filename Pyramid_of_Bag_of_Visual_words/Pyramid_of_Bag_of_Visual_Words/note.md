
Pyrimad of Bag of Visual Words

1. 用來 image classification (pyrimad 原本是用做 object detection)




步驟
Step #1: Sampling our dataset
    sample_dataset.py

Step #2: Extracting and indexing features
    index_features.py
        (詳細看 featureindexer.py)

Step #3: Clustering features to form a visual vocabulary
    cluster_features.py

Step #4: Constructing the PBOW representation
    extract_pbow.py

Step #5: Train our model
    train_model.py

Step #6: Evaluating our classifier
    test_model.py

