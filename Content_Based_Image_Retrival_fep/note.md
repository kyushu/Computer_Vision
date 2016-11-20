

# Bag of Visual word model

### 用途
1. 用來分類 image 內容
2. 用來建立 Content Base Image Retrieval

### Redis
執行 ~/required-library/redis-3.2.4/src/redis-server

### 建立
1. Feature extraction: 取出 image dataset 裡每個 image 的 feature
    1. index_features.py : 
        DetectAndDescribe: keypoint detection -> feature extraction
        FeatureIndexer   : store "image_id", "index", "feature" into HDF5 database

2. Codebook construction : 將步驟 1. Feature extraction 的所有 feature 做分類
    1. cluster_features.py:
        ir.vocabulary.py: 將 feature 以 k-means clustering 做分類建立 image 的 vocabulary 並得到 N 個 cluster centroid 代表每個 cluster 的中心點

3. Vector quantization : 將 image dataset 裡的 image 的 features 去跟 vocabulary 的 feature vector (cluster centroid) 做 minimum distance compare, 看這個 image 的每一個 feature 是屬於那個 cluster 的，並紀錄這個 image's feature 屬於 cluster 的次數 
    1. BagOfVisualWords : 將 target image 的 feature vectors 與 vocab 裡的 feature vectors 做 clustering 並返回量化的結果
    2. BOVWIndexer : 儲存量化的結果

1. Extract SIFT (or other local invariant feature vectors) from a dataset of images.
2. Cluster the SIFT features to form a “codebook”.
3. Quantize the SIFT features from each image into a histogram that counts the number of times each “visual word” appears.
