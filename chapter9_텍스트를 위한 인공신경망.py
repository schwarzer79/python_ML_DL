### 텍스트를 위한 인공신경망
"""
딥러닝을 통한 댓글 분석으로 고객들의 호불호를 파악 -> 순환 신경망을 사용

*** 순차 데이터 (Sequential Data) ***
- 텍스트나 시계열 데이터(time seires data)와 같이 순서에 의미가 있는 데이터 / 지금까지의 데이터는 순서와 상관이 없엇음
- 텍스트 데이터 또한 단어의 배열 순서가 중요한 순차 데이터 -> 신경망에 주입 시 순서를 유지해야함
- 때문에 순차 데이터를 이용할 때는 이전에 입력한 데이터를 기억하는 기능이 필요 -> 이전에 학습한 완전 연결 신경망이나 합성곱 신경망은 이러한 기억장치가 없음
 이렇게 입력 데이터 흐름이 앞으로만 이루어지는 신경망 = 피드포워드 신경망(feedforward neural network)
- 이전 데이터가 신경망 층에 순환되도록 하는 신경망 = 순환 신경망(recurrent neural network)

*** 순환 신경망(recurrent neural network) ***
- 일반적인 완전 연결 신경망과 유사 -> 완전 연결 신경망에 이전 데이터 처리 흐름을 순환하는 고리 하나만 추가하면 됨
- 밀집층 뉴런의 출력이 다시 자기자신으로 전달
    (Ex) input(A,B,C) / A -> result_A -> B -> result_B -> ... / input의 결과가 다시 뉴런으로 돌아가 다음 input 결과에 영향을 줌
- 이렇게 sample에 대한 순환을 반복할 떄, 샘플을 처리하는 한 단계를 timestep 이라고 함
- 순환신경망은 이전 tiemstep의 샘ㅍㄹ을 기억하지만 timestep이 오래될수록 순환되는 정보가 희석되어 희미해짐
- 순환신경망에서는 층을 cell이라고 함 / 한 cell에는 여러 뉴런이 있지만 이전 신경망과 달리 뉴런을 모두 표시하지 않고 하나의 셀로 층을 표현 / cell의 출력을 은닉 상태(hidden state)라고 함
- 일반적으로 은닉층의 activation function으로 tanh^2가 많이 사용됨, -1 ~ 1사이의 범위를 가짐 (sigmoid는 0 ~ 1)
- 표기 상 그림에 활성화 함수를 안넣을 수도 있는 데, 표기에만 없는 것이지 활성화 함수 필요
- 피드포워드 신경망과 마찬가지로 순환신경망도 입력과 가중치를 곱하는 데, 순환 신경망은 가중치가 하나 더 존재 -> 이전 tiemstep의 은닉 상태에 곱해지는 가중치
- cell에서는 입력과 이전 timestep의 hidden state를 이용해서 현재 timestep의 hiddenstate를 만듬
- 처음의 timestep은 이전 timestep이 존재하지 않으므로 이전 은닉 상태의 값이 0이라고 가정하고 진행

*** 셀의 가중치와 입출력 ***
- 입력 특성이 4개, 순환층 뉴런이 3개인 경우를 가정 / 완전 연결 신경망과 같이 입력층과 순환층의 뉴런이 모두 연결되기에 가중치 W_x 의 개수는 4*3=12개
- 다음 timestep에 사용되는 hidden state를 위한 가중치 W_h 의 크기는 ?
 -> 각 뉴런의 hidden state는 자기자신을 포함한 모든 뉴런에 전달되므로 순환층 뉴런이 3개라면 가중치 W_h 의 개수는 3*3 = 9개
- 모델 파라미터의 개수 = W_x + W_h + 절편 = 12 + 9 + 3 = 24개 / cell로 표현하는 이유는 hidden state가 모든 뉴런에 순환되기 떄문에 그림으로 표현하기가 어렵기 때문

- 이전 장에서 합성곱 층의 입력은 하나의 sample이 3개의 차원을 가짐 (너비, 높이, 채널) -> 합성곱 층과 pooling 층을 통과하면 그 값은 변하지만 차원 개수는 유지
- 순환층은 sample마다 2개의 차원을 가짐 / 하나의 sample을 sequence라고 함 / sequence의 길이 = timestep의 길이
- 하나의 sample은 시퀀스 길이(단어 개수)와 단어 표현의 2차원 배열로 구성됨 -> 순환층을 통과하면 1차원 배열로 변환 (크기는 순환층의 뉴런 개수에 의해 결정)
- 순환층은 마지막 timestep의 은닉상태만을 출력으로 내보냄 = 입력 sequence 길이를 모두 읽고 정보를 마지막 hidden state에 압축해 전달

- 순환신경망도 여러 개의 층을 쌓을 수 있는 데 이 때의 출력은?
 -> cell의 입력은 timestep과 단어 표현으로 이루어진 2차원 배열이어야하므로 첫번째 cell에서 마지막 timestep의 은닉상태만 출력해서는 안됨 -> 마지막 셀을 제외한 모든 셀은 모든 타임스텝의 은닉상태 출력
- 다른 신경망과 마찬가지로 순환 신경망도 마지막 층에는 밀집층을 두어 클래스를 분류
 -> 다중 분류인 경우는 출력층에 클래스 개수만큼 뉴런을 두고 softmax 함수 사용
 -> 이진 분류인 경우에는 하나의 뉴런을 두고 sigmoid 함수 사용
- 합성곱과 다른 점은 마지막 cell의 출력이 1차원 배열이므로 Flatten()을 이용해서 펼칠 필요가 없다는 것
 (ex) input.shape = (20,100) 인 경우, 20개의 timestep과 100개의 특성으로 구성 / 뉴런 10개로 구성된 순환층을 지나가면 (10,) 출력
"""

### 순환신경망으로 IMDB 리뷰 분류하기

## IMDB 리뷰 데이터셋
"""
자연어 처리(NLP) : 컴퓨터를 사용해 인간의 언어를 처리하는 분야 (음성 인식, 기계 번역, 감성 분석 등)
말뭉치(corpus) : 자연어 처리 분야에서 훈련 데이터를 지칭

텍스트를 분석하기 위해서 순환신경망에 데이터를 전달할 때, 텍스트 그 자체를 그대로 전달하지는 않음
- 이미지의 경우 구성하고 있는 픽셀이 숫자이기에 별다른 변환이 필요없었지만 텍스트의 경우 변환이 필요 -> 일반적인 방법은 등장하는 단어마다 고유 정수를 부여하는 것
- 정수값 사이에는 별다른 관계가 없고 단지 구분하는 역할 / 영어 문장의 경우 소문자로 모두 바꾸고 구둣점을 삭제하고 공백 기준 분리 --> 분리된 단어 = 토큰(token)
- 하나의 샘플이 여러 token으로 이루어져 있고 1개의 토큰은 하나의 timestep에 해당

* 어휘사전 - 훈련 세트에서 고유한 단어를 뽑아서 만든 목록
"""

from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words = 500)
print(train_input.shape, test_input.shape) # (25000,) (25000,) / 각 텍스트의 길이가 제각각이기에 크기가 고정된 2차원 배열보다 메모리를 효율적으로 사용하기 위해 1차원 배열 이용
print(len(train_input[0])) # 218 -> 218개의 토큰으로 구성
print(len(train_input[1])) # 189

print(train_input[0]) # 첫번째 리뷰에 담긴 내용 출력(정수) / IMDB 리뷰데이터는 이미 정수로 변환되어 있는 상태  / 500개의 단어는 많이 나오는 순서대로 선택한 것

print(train_target[:20]) # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1] / 타깃데이터 (0 = 부정, 1 = 긍정)

# validation set
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state=42)

# 각 리뷰의 길이를 확인하기 위한 numpy array
import numpy as np
lengths = np.array([len(x) for x in train_input])
print(np.mean(lengths), np.median(lengths)) # 239.00925 178.0 -> 한쪽에 치우친 분포

import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show(block=True) # 대부분의 리뷰 길이는 300미만 / 리뷰가 대체로 짧아 중간값보다 훨씬 작은 100개의 단어를 사용, 100개보다 작은 리뷰를 길이 100에 맞추기 위해 padding / padding 토큰은 0

# pad_sequences()를 이용한 시퀀스 데이터 길이 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen = 100) # maxlen에 원하는 길이를 지정하면 긴 경우에는 잘라내고 짧으면 0으로 패딩
print(train_seq.shape) # (20000, 100) -> train_input은 파이썬 리스트 배열이었지만 100으로 맞춰 train_seq는 2차원 배열이 됨

print(train_seq[0]) # pad_sequences에서 maxlen보다 긴 경우 뒤가 아닌 앞에서부터 자름 / 앞부분 정보보다 뒷부분 정보가 더 유용할 거 같다는 판단에 의한 것 / 뒷부분을 잘라내고 싶다면 truncating = 'post'로 변경
print(train_seq[5]) # 0이 있는 것으로 보아 원래는 maxlen < 100

val_seq = pad_sequences(val_input, maxlen = 100)

## 순환 신경망 만들기
"""
keras에서는 여러 가지 종류의 순환층 클래스를 제공 / 가장 간단한 것은 simpleRNN 클래스
"""

from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(8, input_shape = (100,500))) # 샘플 길이 100 / activation 기본값이 tanh이므로 표기안함
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

"""
input_shape의 두번째 차원이 500인 이유?
- 이전에 pad_sequences로 만든 train_seq나 val_seq는 문제가 있음 -> 토큰을 정수로 변환한 데이터를 신경망에 주입하면 큰 정수가 큰 활성화 출력을 생성함
- 하지만 토큰의 정수값 사이에는 아무런 영향이 없어야하므로 다른 방식의 입력을 찾아야 함
- 정수값에 있는 크기 속성을 없애고 각 정수를 고유하게 표현하는 방법 = One - Hot Encoding -> 해당 위치의 원소만 1이고 나머지는 0으로 변환
- 이전 imdb.load_data()에서 500개 단어만 사용하도록 지정했기에 고유 단어는 500개 -> 훈련 데이터 정수값의 범위가 0 ~ 499까지이므로 원-핫 인코딩으로 표현하려면 배열 길이가 500이어야 함
- 각 숫자를 500개의 칸을 이용한 one-hot encoding으로 표현하는 것
    이를 위해 keras.utils.to_categorical() 함수를 이용할 수 있음
"""

train_oh = keras.utils.to_categorical(train_seq)
print(train_oh.shape) # (20000, 100, 500) / 본래 입력데이터는 (20000,100) 이었으나 각 토큰이 500차원의 배열로 변경되어 (20000, 100, 500)으로 변경
# 이렇게 sample 데이터가 (100,) 에서 (100,500)으로 바꿔야 하므로 input_shape = (100, 500)으로 설정

print(train_oh[0][0][:12]) # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
print(np.sum(train_oh[0][0])) # 1.0

val_oh = keras.utils.to_categorical(val_seq)
model.summary()
"""
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 8)                 4072    # 4000(500차원의 one-hot encoding * 순환층 뉴런 8) + 64 (순환층 뉴런 개수 * 은닉 상태 크기) + 8 (절편) = 4072
                                                                 
 dense (Dense)               (None, 1)                 9         
                                                                 
=================================================================
Total params: 4,081
Trainable params: 4,081
Non-trainable params: 0
_________________________________________________________________
"""

## 순환신경망 훈련하기 - keras API를 이용하기에 기존 신경망 훈련과 다르지 않음

rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoints_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights= True)
history = model.fit(train_oh, train_target, epochs = 100, batch_size = 24, validation_data = (val_oh, val_target), callbacks = [checkpoints_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block = True)

"""
one-hot encoding을 이용해서 입력 데이터의 정수 크기 속성을 삭제했지만 이렇게 하면 입력 데이터의 크기가 매우 커지는 단점이 있음
"""

## 단어 임베딩 사용
"""
텍스트 처리 시 word embedding을 사용하면 각 단어를 고정된 크기의 실수 벡터로 바꿔줌 -> one-hot encoding보다 더 의미있는 값으로 채워져 자연어 처리에서 더 좋은 성능을 가짐
keras.layers.Embedding 클래스를 층으로 넣으면 모든 벡터가 랜덤하게 초기화되지만 훈련을 통해 좋은 단어 임베딩을 학습
단어 임베딩은 입력을 정수로 받기 때문에 메모리를 더 효율적으로 사용할 수 있음
"""

model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length = 100)) # 500은 어휘사전의 크기, 16은 임베딩 벡터의 크기(one-hot encoding보다 훨씬 작음), input_length는 샘플 길이 = 100
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
model2.summary()
"""
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 100, 16)           8000      # 500개의 word를 16크기의 벡터로 변경하기 때문에 8,000개의 모델 파라미터를 가짐
                                                                 
 simple_rnn_1 (SimpleRNN)    (None, 8)                 200      # 16(임베딩 벡터 크기) * 8(뉴런 개수) + 8 * 8(은닉상태 가중치) + 8(절편) = 200
                                                                 
 dense_1 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,209
Trainable params: 8,209
Non-trainable params: 0
_________________________________________________________________
"""

rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model2.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoints_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights= True)
history = model2.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoints_cb, early_stopping_cb]) # epoch = 30

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block = True)


### LSTM과 GRU 셀
"""
simpleRNN보다 계산이 훨씬 복잡하지만 성능이 뛰어나 순환신경망에서 많이 채택
기본적인 순환층은 긴 시퀀스를 학습하기 어려워짐 -> 길어질수록 순환되는 hidden state에 담긴 정보가 점차 희석되기 때문에 멀리 떨어져 있는 단어 정보를 인식하는 것이 어려웠음 --> LSTM + GRU 셀

*** LSTM 구조 ***
LSTM (Long Short - Term Memory) 는 단기 기억을 오래가도록 하기 위해서 고안 / 구조는 복잡하지만 기본 골자는 비슷
입력과 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러 개 사용하고 이 결과를 다음 timestep에서 재사용

* hidden state를 만드는 방법
hidden state = 입력과 이전 timestep의 hidden state를 가중치에 곱한 후 활성화 함수를 통과시켜 다음 hidden state를 만들어냄, 이 떄 기존 순환층과는 다르게 sigmoid 함수를 사용하고, tanh 활성화 함수를 통과한 어떤 값과 곱해져서 hidden state를 만듦
그렇다면 tanh를 통과한 값이 무엇인가?
- LSTM에서는 순환되는 상태가 2개있음 = hidden state + cell state / cell state = 다음 층으로 전달되지 않고 LSTM 셀에서 순환만 되는 값 / CELL STATE에 관한 내용은 교재 참조
"""

## LSTM 신경망 훈련하기
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words = 500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state = 42)

# pad_sequences()를 이용한 sample 길이 맞추기
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen = 100)
val_seq = pad_sequences(val_input, maxlen = 100)

# LSTM 셀을 이용한 순환층 만들기
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length = 100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
model.summary()
"""
Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, 100, 16)           8000      
                                                                 
 lstm (LSTM)                 (None, 8)                 800       # simpleRNN에서는 모델 파라미터가 200개였으나 LSTM은 내부에 작은 셀 4개가 더 있으므로 800개
                                                                 
 dense_4 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,809
Trainable params: 8,809
Non-trainable params: 0
_________________________________________________________________
"""

rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoints_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)
history = model.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoints_cb, early_stopping_cb])

# 그래프 그리기
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block =True)

## 순환층에 드롭아웃 적용하기 - 과대적합 방지
"""
이전 완전 연결 신경망이나 합성곱 신경망에서는 Dropout 클래스를 사용해 드롭아웃을 적용
순환층은 자체적으로 드롭아웃 기능을 제공 -> SimpleRnn과 LSTM 클래스 모두 dropout 매개변수(셀의 입력에 적용)와 recurrent_dropout 매개변수(순환되는 hidden state에 적용)를 가지고 있음 / recurrent_dropout을 사용하면 GPU를 이용한 모델 훈련이 불가능
"""

model2 =keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length = 100))
model2.add(keras.layers.LSTM(8, dropout = 0.3))
model2.add(keras.layers.Dense(1, activation = 'sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model2.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoints_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoints_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block =True)

## 2개의 층을 연결하기
"""
순환층의 hidden state는 마지막 timestep의 결과만을 다음 층으로 전달하는 데, 순환층이 중첩되어 있는 경우 뒤에 오는 순환층의 입력은 순차 데이터이어야 하므로 앞쪽 순환층은 모든 timestep에 대한 hidden state를 출력해야함
    -> return_sequences = True 로 지정
"""

model3 =keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length = 100))
model3.add(keras.layers.LSTM(8, dropout = 0.3, return_sequences = True))
model3.add(keras.layers.LSTM(8, dropout = 0.3))
model3.add(keras.layers.Dense(1, activation = 'sigmoid'))

model3.summary()
"""
Model: "sequential_8"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_6 (Embedding)     (None, 100, 16)           8000      
                                                                 
 lstm_4 (LSTM)               (None, 100, 8)            800       
                                                                 
 lstm_5 (LSTM)               (None, 8)                 544       
                                                                 
 dense_7 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 9,353
Trainable params: 9,353
Non-trainable params: 0
_________________________________________________________________
"""
rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model3.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoints_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)
history = model3.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoints_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block =True)

## GRU 구조
"""
GRU (Gated Recurrent Unit) : LSTM의 간소화 버전으로 LSTM처럼 cell state를 계산하지 않고 hidden state 하나만 포함 
- GRU 셀에는 hiddens state와 input에 가중치를 곱하고 절편을 더하는 작은 셀이 3개 들어 있음 (2개는 sigmoid, 1개는 tanh 활성화 함수 사용)
- 가중치 W_z를 사용하는 셀의 출력이 hidden state에 바로 곱해져 삭제 게이트 역할을 수행
- W_z에서 나온 출력을 1에서 뺸 후 가장 오른쪽 W_g를 사용하는 셀의 출력에 곱함 = 입력 정보 제어 역할
- W_r를 사용하는 셀에서 나온 출력은 hidden state의 정보를 제어
- GRU는 LSTM보다 가중치가 적어 계산량이 작지만 LSTM 못지않은 성능을 발휘
"""

## GRU 신경망 훈련하기
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length = 100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation = 'sigmoid'))

model4.summary()
"""
Model: "sequential_10"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_7 (Embedding)     (None, 100, 16)           8000      
                                                                 
 gru (GRU)                   (None, 8)                 624       #  16*8(입력 가중치) + 8*8(hidden state 가중치) + 8 = 200 * 3 (GRU 내부 셀 개수) / 624인 이유는 tensorflow에서 사용하는 GRU 셀이 다른 형태이기 때문
                                                                 
 dense_8 (Dense)             (None, 1)                 9         
                                                                 
=================================================================
Total params: 8,633
Trainable params: 8,633
Non-trainable params: 0
_________________________________________________________________
"""

rmsprop = keras.optimizers.RMSprop(learning_rate = 1e-4)
model4.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics = ['accuracy'])
checkpoints_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5', save_best_only = True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience = 3, restore_best_weights=True)
history = model4.fit(train_seq, train_target, epochs = 100, batch_size = 64, validation_data = (val_seq, val_target), callbacks = [checkpoints_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show(block =True)

test_seq = pad_sequences(test_input, maxlen = 100)
rnn_model = keras.models.load_model('best-2rnn-model.h5')
rnn_model.evaluate(test_seq, test_target) # [0.43067148327827454, 0.8000400066375732]
