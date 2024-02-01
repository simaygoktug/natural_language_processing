#Veri ön işleme için standartlaştırma yapılır:
# 1) Metinleri küçük harfe çevirme.
# 2) Noktalama işaretlerini kaldırma.

#Bir diğer veri ön işleme adımı stemming'dir. Kelimelerin sadece kökleriyle çalışmayı sağlar.

#Daha sonra metinler token denilen en küçük parçalara ayrılır. 
#Tokenler indexlenerek vektörleştirilir.
#Tokenleştirme 2 farklı yöntem ile yapılır: 
# 1) Kelime bazlı --> Boşluklara göre ayırma.
# 2) n-gram --> Harf sayısına göre ayırma. Kelimelerin sırasına dikkat edilmez.
#Her indisteki kelime one-hot encoding ile encode edilir. 

#Genelde en sık kullanılan 20.000 kelimelik vocabulary tercih edilir yoksa model çok şişer.
#Sözlükte olmayan bir kelime ile karşılaşıldığında index1 ile gösterilir.
#0 uygulanmaz çünkü o bilgi kaybolmasın diye kullandığımız padding tekniğinde kullanılır.
#Örneğin uzunlukları aynı olmayan iki diziden eksik olana 0 ekleriz ki uzunlukları eşitlensin ve kolay hesaplama yapılsın diye.

#TextVectorization katmanı tüm bu veri ön işleme adımlarında kolaylık sağlar.

import tensorflow as tf 
from tensorflow import keras 
from keras import layers, models, optimizers, losses, metrics
from keras.layers import TextVectorization  

text_vectorization=TextVectorization()
data=[
    "I love computer vision projects",
    "You can not finish a natural language processing task"
    "I will work at META"
]
text_vectorization.adapt(data)
print(text_vectorization.get_vocabulary())
vectorized_text=text_vectorization(data)
print(vectorized_text)

#Kendi standartlaştırma fonksiyonumuzu yazabiliriz:

import re
import string

def standardization_fn(string_tensor):
  lowercase=tf.strings.lower(string_tensor)
  return tf.strings.regex_replace(
      lowercase, f"[{re.escape(string.punctuation)}]", ""
  )

def split_fn(string_tensor):
  return tf.strings.split(string_tensor)

text_vectorization = TextVectorization(
    standardize=standardization_fn,
    split = split_fn
)

text_vectorization.adapt(data)

text = "bugün ece çok güzel"
text_vectorization(text)

text_dataset = tf.data.Dataset.from_tensor_slices([
    "kedi", "aslan", "yunus"
])

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_sequence_length=4
)

vectorize_layer.adapt(text_dataset.batch(64))

vectorize_layer.get_vocabulary()

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,), dtype=tf.string),
    vectorize_layer
])

input_data=[["kedi kartal aslan"], ["fok yunus"]]

model.predict(input_data)
