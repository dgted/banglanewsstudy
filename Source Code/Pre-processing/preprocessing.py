import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns
import re,nltk,json, pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM,GRU
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
np.random.seed(42)
class color: # Text style
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
# Warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# dataset path

data = pd.read_csv('datasetittefaq.csv',encoding='utf-8')
data=data.dropna()
print(f'Total number of Documents: {len(data)}')
dfs=data
a=['budget','culture','spotlight','lifestyle','bookfair','election','print-edition','travel']
for i in a:
    dfs=dfs[dfs.type !=i ]
    
data=dfs

# Plot the Class distribution
sns.set(font_scale=1.4)
data['type'].value_counts().plot(kind='barh', figsize=(8, 12))
plt.ylabel("Number of Articles", labelpad=12)
plt.xlabel("Category", labelpad=12)
plt.xticks(rotation = 45)
plt.title("Article Summary", y=1.02);

def cleaning_documents(articles):

      news = articles.replace('\n',' ')
      news = re.sub('[^\u0980-\u09FF]',' ',str(news))
 
      stp = open('/content/drive/MyDrive/bangla_stopwords.pkl','r', encoding= 'unicode_escape').read().split()
      result = news.split()
      news = [word.strip() for word in result if word not in stp ]
      news =" ".join(news)
      return news

data['cleaned'] = data['text'].apply(cleaning_documents) 


def data_summary(dataset):

  documents = []
  words = []
  u_words = []

  class_label = [k for k,v in dataset.type.value_counts().to_dict().items()]
  for label in class_label: 
    word_list = [word.strip().lower() for t in list(dataset[dataset.type==label].cleaned) for word in t.strip().split()]
    counts = dict()
    for word in word_list:
      counts[word] = counts.get(word, 0)+1
  
    ordered = sorted(counts.items(), key= lambda item: item[1],reverse = True)

    documents.append(len(list(dataset[dataset.type==label].cleaned)))

    words.append(len(word_list))

    u_words.append(len(np.unique(word_list)))
       
    print("\nClass Name : ",label)
    print("Number of Documents:{}".format(len(list(dataset[dataset.type==label].cleaned))))  
    print("Number of Words:{}".format(len(word_list))) 
    print("Number of Unique Words:{}".format(len(np.unique(word_list)))) 
    print("Most Frequent Words:\n")
    for k,v in ordered[:10]:
      print("{}\t{}".format(k,v))
  return documents,words,u_words,class_label


documents,words,u_words,class_names = data_summary(dataset)


data_matrix = pd.DataFrame({'Total Documents':documents,
                            'Total Words':words,
                            'Unique Words':u_words,
                            'Class Names':class_names})
data_matrix