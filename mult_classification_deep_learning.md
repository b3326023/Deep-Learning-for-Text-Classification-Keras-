

```python
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
import numpy as np
import jieba
```


```python
def loadRawFile(rawFile_Path = ""):
        ret_raw_set = []

        for line in open(rawFile_Path,"r",encoding="utf-8"):
            block=[line.rstrip()[0:9],line.rstrip()[10:16],line.rstrip()[17:21],line.rstrip()[22:]]
            ret_raw_set.append(block)
        
        return ret_raw_set
```


```python
def loadFile(trainSet_Path = ""):
        ret = []
        with open(trainSet_Path, 'r',encoding='utf-8') as f:
            for line in f:
                line=line.replace(' ','')
                line=line.replace('[','')
                line=line.replace(']','')
                line=line.replace("'",'')
                line=line.strip()
                line=line.split(',')
                ret.append(line)
        return ret
```


```python
def segment(train_set = [],user_dict_Path="",user_stopwords_Path="",jieba_dict_Path=""):
        import jieba
        import codecs
        from bs4 import BeautifulSoup
        import unicodedata
        jieba.load_userdict(jieba_dict_Path) 
        jieba.load_userdict(user_dict_Path) 

        stopwords = codecs.open(user_stopwords_Path,"r",encoding="utf8").readlines()    #打開停用字檔案
        stopwords = [ w.strip() for w in stopwords ]        #去除停用字空白

        ret_train_set = []
        for line in train_set:
            content = BeautifulSoup(line,"html.parser").get_text()
            content=unicodedata.normalize("NFKD", content)          #將/xa0換成空白
            ret_train_set.append([w for w in list(jieba.cut(content,cut_all=False)) if w not in stopwords])

        return ret_train_set
    
```


```python
#標籤資料原本包含eventNo,eventCode,SAC，用此function取得eventCode
def getEventCode(y):
    ret = []
    for code in y:
        ret.append(str(code[1].strip("\"")))
    return ret
```


```python
#eventCode本有11類，用此function可以選擇某一類事件類別，做二元劃分。如:輸入0102，則劃分後僅會有"跌倒事件"和"非跌倒事件"
def toBinaryEventCode(eventCode,code):
    ret=eventCode.copy()
    for i,x in enumerate(eventCode):
        if x == code:
            ret[i] = 1
        else:
            ret[i] = 0
    return ret
```


```python
#應 Keras 所需，將斷詞 List之間插入一個空格
def insertSpase(train_x):
    Train_x = []
    temp = ""
    for x in train_x:
        for y in x:
            temp += ' ' + y
        Train_x.append(temp)
        temp = ""
    return Train_x
```


```python
#讀取原始檔，包括文本資料 and 標籤(包含eventNo,eventCode,SAC)
train_x = loadFile('D:/Python/高榮專案/deep learning/data/description_processed.txt')
train_y = loadFile('D:/Python/高榮專案/deep learning/data/head_processed.txt')
```


```python
#取得所有資料的eventCode
eventCode = getEventCode(train_y)
```


```python
#將eventCode化為 one-hot格式，共11類
code, num_eventCode = np.unique(eventCode,return_inverse=True)
OneHot_eventCode = np_utils.to_categorical(num_eventCode,num_classes=11)
```


```python
#做二元劃分
#binary_eventCode = toBinaryEventCode(eventCode,'0102')
```

    ['0102', '0109', '0102', '0109', '0105', '0102', '0109', '0101', '0102', '0102']
    [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
    


```python
#應 Keras 所需，將斷詞 List之間插入一個空格
Train_x = insertSpase(train_x)
```


```python
#建立斷詞器
token = Tokenizer(num_words = 10000)
token.fit_on_texts(Train_x)
```


```python
#使用斷詞器將文數字化
Train_x_seq = token.texts_to_sequences(Train_x)  #將文本轉為數字List
Train_x_ok = sequence.pad_sequences(Train_x_seq,maxlen=100)  #截長補短文本數字List至長度100
```


```python
#建立多分類模型，包含嵌入層、平坦層、一個隱藏層、輸出層，輸出層僅有十一個神經元，代表十一個事件類別，使用softmax函數，能識別十一個類別。
mul_model = Sequential()
mul_model.add(Embedding(output_dim=32,input_dim=10000,input_length=100))
mul_model.add(Dropout(0.2))
mul_model.add(Flatten())
mul_model.add(Dense(units=256,activation='relu'))
mul_model.add(Dropout(0.35))
mul_model.add(Dense(units=11,activation='softmax'))
mul_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#使用OneHot_eventCode作為目標值 (多種類別)
train_history = mul_model.fit(Train_x_ok,OneHot_eventCode,batch_size=100,epochs=5,verbose=2,validation_split=0.2)
#預測訓練集的結果
predict = mul_model.predict_classes(Train_x_ok)
predict = predict.reshape(-1)
```

    Train on 5875 samples, validate on 1469 samples
    Epoch 1/5
     - 3s - loss: 0.2311 - acc: 0.9213 - val_loss: 0.1798 - val_acc: 0.9411
    Epoch 2/5
     - 2s - loss: 0.1273 - acc: 0.9571 - val_loss: 0.0953 - val_acc: 0.9660
    Epoch 3/5
     - 2s - loss: 0.0626 - acc: 0.9796 - val_loss: 0.0690 - val_acc: 0.9746
    Epoch 4/5
     - 2s - loss: 0.0339 - acc: 0.9899 - val_loss: 0.0627 - val_acc: 0.9764
    Epoch 5/5
     - 2s - loss: 0.0180 - acc: 0.9948 - val_loss: 0.0618 - val_acc: 0.9774
    


```python

#隨意使用一筆測試資料做預測，一樣先轉為數字List，截長補短，接著丟入模型預測結果。
inputData = ' 開錯 劑量 藥局 藥物 藥師 劑量'
input_seq = token.texts_to_sequences([inputData])
input_ok = sequence.pad_sequences(input_seq,maxlen=100)
# print(input_ok)
predict_input = model.predict_classes(input_ok)
predict_input = predict_input.reshape(-1)
print('預測: 第',predict_input[0],'類')
```

    預測: 第 0 類
    


```python
# 建立二元分類模型，包含嵌入層、平坦層、一個隱藏層、輸出層，輸出層僅有一個神經元，使用sigmoid函數，僅能判別二元分類
# bi_model = Sequential()
# bi_model.add(Embedding(output_dim=32,input_dim=10000,input_length=100))
# bi_model.add(Dropout(0.2))
# bi_model.add(Flatten())
# bi_model.add(Dense(units=256,activation='relu'))
# bi_model.add(Dropout(0.35))
# bi_model.add(Dense(units=1,activation='sigmoid'))
# bi_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#使用binary_eventCode作為目標值 (只有二種分類，是or不是)
# train_history = bi_model.fit(Train_x_ok,binary_eventCode,batch_size=100,epochs=5,verbose=2,validation_split=0.2)

#預測訓練集的結果
# predict = bi_model.predict_classes(Train_x_ok)
# predict = predict.reshape(-1)

```


```python
#隨意使用一筆測試資料做預測，一樣先轉為數字List，截長補短，接著丟入模型預測結果。
# inputData = '滑倒 意識 骨盆 輕微 碎裂 脊椎 骨折 膝蓋'
# input_seq = token.texts_to_sequences([inputData])
# input_ok = sequence.pad_sequences(input_seq,maxlen=100)

# predict_input = bi_model.predict_classes(input_ok)
# predict_input = predict_input.reshape(-1)
# print('預測:',predict_input[0])
```
