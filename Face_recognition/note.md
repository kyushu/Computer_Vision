
Usage:
    1. gather_selfies.py : 
        使用 Haar cascade Face detection (cv2.CascadeClassifier(haarcascade_frontalface_default.xml))
        擷取 camera 上偵測到的 "人臉" 
        並將 image data 轉成 base64字串，每個base64字串代表一張人臉影像再存入對應名稱的檔案
        例如 morpheus.txt

    2. train_recognizer.py : 
        使用 LBP (cv2.createLBPHFaceRecognizer) 作為 classifier

    3. recognize.py : 
        從 camera 或 poto 辨識 人臉

    
