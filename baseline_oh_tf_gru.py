
# %%
import pandas as pd
import gensim
import numpy as np
import warnings
import os
from sklearn.preprocessing import LabelEncoder

# 开启GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
warnings.filterwarnings('ignore')

# 内存压缩读取
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# train_path = '../data/train.txt'
# test_path = '../data/apply_new.txt'
train_path = '../../data/train.txt'
test_path = '../../data/apply_new.txt'

# %%

def getRawData(train,test):
    # 读取数据集
    train = (pd.read_csv(train, header=None))
    test = (pd.read_csv(test, header=None))
    train.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
    test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
    train['label'] = train['label'].astype(int)
    data = pd.concat([train,test])
    data['label'] = data['label'].fillna(-1)
    data['tagid'] = data['tagid'].apply(lambda x:eval(x))
    data['tagid'] = data['tagid'].apply(lambda x:[str(i) for i in x])
    data['time'] = data['time'].apply(lambda x:eval(x))
    data['time'] = data['time'].apply(lambda x:[str(i) for i in x])
    data['gender'].fillna(2,inplace=True)
    data['age'].fillna(0,inplace=True)
    # 对province,city,model,make进行标签分配
    columns = ['province','city','model','make']
    for column in columns:
        tmp = pd.DataFrame({'column' : data[column]})
        le = LabelEncoder()
        le.fit(np.unique(tmp['column'].values))
        data[column] = pd.DataFrame(le.transform(data[column].values),columns=[column])

    for column in ['gender','age']:
        data[column] = data[column].astype(int)
    # 数据分布分析 首先处理空值 并对数据进行标签化
    X_train = data[ data['label'] != -1]
    X_test = data[data['label'] == -1]
    y = X_train['label']
    return X_train,y,X_test


# %%
# 加载数据
train,y,test = getRawData(train_path,test_path)
print('数据读取完成')
# %%
# 重点只关注 年龄 省份 城市 三个基本特征
unimportant_columns = ['gender','model','make']
train = train[[x for x in train.columns if x not in unimportant_columns]]
test = test[[x for x in test.columns if x not in unimportant_columns]]
data = pd.concat([train,test])

'''
Onehot encoding
'''
from sklearn import preprocessing
enc = preprocessing.LabelBinarizer()
important_columns = ['age','province','city'] 
one_hot_features = []
for col in important_columns:
    one_hot = enc.fit_transform(data[col])
    one_hot_features.append(one_hot)
nf = np.hstack([(oh) for oh in one_hot_features])


# %%
# 在这里需要重新将one hot编码后的数字拼接到tagid前面
seq = data['tagid'].to_list()

# %%
input_feature = list()
for idx in range(0,nf.shape[0]):
    input_feature.append(list(nf[idx]) + seq[idx])

# %%
print('开始组合特征')
# 重新组织 基本信息 + 序列数据 的特征
# 利用CountVector和TFIDF提取特征
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy import sparse

print('开始组合Ti-IDF特征')
'''数据处理'''
def get_new_sequence(x):
    x = list(x)
    arr = list()
    for d in x:
        arr.append(str(d).strip('[').strip(']').replace(',',' '))
    return pd.DataFrame({'new_review' : arr})
new_review = get_new_sequence(data['tagid'])
# tidf读取特征
tfidfVec = TfidfVectorizer( stop_words=ENGLISH_STOP_WORDS,ngram_range=(1,1),max_features=64)
tfidfVec.fit(new_review['new_review'])
data_cat = tfidfVec.transform(new_review['new_review'])
df_tfidf = pd.DataFrame(data_cat.toarray())
df_tfidf.columns = ['tfidf_' + str(i) for i in df_tfidf.columns]


# %%
'''
将tf-idf提取的特征加入到input_features前面 重新组合为新的输入特征
在这里要注意输入数据类型的一致
'''
input_data = list()
tmp = np.array(df_tfidf)
for idx in range(0,len(input_feature)):
    input_data.append(list(tmp[idx]) + input_feature[idx])


# %%
# nf是重新编码过后的特征 接下来分别使用两种方式去构建模型
'''
开始word embedding过程 输入到网络层 这里的时间训练时间较长
'''
print('开始embedding')
from gensim.models import FastText
from tensorflow.keras.preprocessing import text, sequence
embed_size = 64
MAX_NB_WORDS = 230637
MAX_SEQUENCE_LENGTH = 128
# 这里换成fastext
f2v_model = FastText(sentences=input_feature, vector_size=embed_size, window=5, min_count=1,epochs=5)
f2v_model.wv.save_word2vec_format("model/f2Vec_60" + ".bin", binary=True)
print('模型保存成功')
# %%
#f2v_model = Word2Vec.load('../model/f2v_onehot_idf_seq.m')

new_data = pd.DataFrame({"tagid" : input_data})
# %%
def int32ToStr(col):
    col = [str(c) for c in col]
    return col
new_data['tagid'].apply(int32ToStr)


# %%
X_train = new_data[:train.shape[0]]['tagid'].apply(lambda x:[str(i) for i in x])
X_test = new_data[train.shape[0]:]['tagid'].apply(lambda x:[str(i) for i in x])


# %%
# 创建词典，利用了tf.keras的API，其实就是编码一下，具体可以看看API的使用方法
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


# %%
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
# 计算一共出现了多少个单词，其实MAX_NB_WORDS我直接就用了这个数据
nb_words = len(word_index) + 1
print('Total %s word vectors.' % nb_words)
# 构建一个embedding的矩阵，之后输入到模型使用
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    try:
        embedding_vector = f2v_model.wv.get_vector(word)
    except KeyError:
        continue
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

y_categorical = train['label'].values


# %%
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from attention import Attention as at
def my_model():
    # 词嵌入（使用预训练的词向量）
    embedding_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    '''
    input_dim : nb_words 代表单词表单词的个数加1
    output_dim : embed_size 代表输出的维度
    input_length : 输出的长度
    '''
    embedder = Embedding(nb_words,
                         embed_size,
                         input_length=MAX_SEQUENCE_LENGTH,
                         weights=[embedding_matrix],
                         trainable=False
                         )
    embed = embedder(embedding_input)
    l = GRU(512,return_sequences=True)(embed) # 256 is the best
    l = at()(l)
    flat = BatchNormalization()(l)
    drop = Dropout(0.15)(flat) # 改变学习率
    main_output = Dense(1, activation='sigmoid')(drop)
    model = Model(inputs=embedding_input, outputs=main_output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['accuracy'])
    return model


# %%
from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import StratifiedKFold
# 五折交叉验证
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros([len(train), 1])
predictions = np.zeros([len(test), 1])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print("fold n{}".format(fold_ + 1))
    model = my_model()
    if fold_ == 0:
        model.summary()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
    bst_model_path = "./{}keras_GRU_所有特征_gru.h7".format(fold_)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    X_tra, X_val = X_train[trn_idx], X_train[val_idx]
    y_tra, y_val = y_categorical[trn_idx], y_categorical[val_idx]

    model.fit(X_tra, y_tra,
              validation_data=(X_val, y_val),
              epochs=128, batch_size=512, shuffle=True,
              callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)

    oof[val_idx] = model.predict(X_val)

    predictions += model.predict(X_test) / folds.n_splits
    print(predictions)
    del model

train['predict'] = oof
train['rank'] = train['predict'].rank()
train['p'] = 1
train.loc[train['rank'] <= train.shape[0] * 0.5, 'p'] = 0
bst_f1_tmp = f1_score(train['label'].values, train['p'].values)
print(bst_f1_tmp)

submit = test[['pid']]
submit['tmp'] = predictions
submit.columns = ['user_id', 'tmp']

submit['rank'] = submit['tmp'].rank()
submit['category_id'] = 1
submit.loc[submit['rank'] <= int(submit.shape[0] * 0.5), 'category_id'] = 0
print(submit['category_id'].mean())
submit[['user_id', 'category_id']].to_csv('open_{}.csv'.format(str(bst_f1_tmp).split('.')[1]), index=False)


