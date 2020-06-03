"""
Created on Wed Jun  3 10:03:07 2020

@author: prbpedro
"""

#gerando as stopwords
import nltk  #biblioteca para a análise de textos
import numpy as np
# nltk.download('stopwords')
# nltk.download('punkt') # Linguagem portuguesa
from pprint import pprint  #biblioteca para realizar o "print" de forma mais amigável "pretty-print"

stopWordPortugues = nltk.corpus.stopwords.words('portuguese')
print(np.transpose(stopWordPortugues))

#gerando os tokens de sentenças
sample_text=""" O menino gosta de jogar futebol aos finais de semana. 
Ele gosta de jogar com seus amigos Marcos e João, mas não gosta de brincar
com a irmã Marcela"""
sample_sentence = nltk.sent_tokenize(text=sample_text)
pprint(sample_sentence)

len(sample_sentence)

#tokenização de palavras
sample_sentence='O menino gosta de jogar futebol aos finais de semana.'
sample_words = nltk.word_tokenize(text=sample_sentence)
pprint(sample_words)

#gerando as amostras de stem
from nltk.stem import PorterStemmer  #stemização baseado no algoritmo de Porter
from nltk.stem import RSLPStemmer #stemização para o português
# nltk.download('rslp')

#gerado stem através do nltk
ps=PorterStemmer()
stemmer=RSLPStemmer()

print(ps.stem('jumping'))
print(stemmer.stem('amoroso')) 
print(stemmer.stem('amorosa'))
print(stemmer.stem('amados'))

from nltk.stem import SnowballStemmer   # mais indicado para a lingua portuguesa

print('Linguagens suportadas %s', SnowballStemmer.languages)

ss = SnowballStemmer("portuguese")
print(ss.stem('casado'))
print(ss.stem('casarão'))
print(ss.stem('casa'))

"""** ---------------------**Exemplo Bag of Words ---------------------**** **"""

sentenca="O IGTI oferece especializacao em Deep Learning. Deep Learning e utilizado em diversas aplicacoes. As aplicacoes de deep learning sao estudadas nesta especializacao. O IGTI tambem oferece bootcamp"

#coloca toda a sentença em lowercase
sentenca=sentenca.lower()

print(sentenca)

#tokenização de sentencas
sample_sentence = nltk.sent_tokenize(text=sentenca)
pprint(sample_sentence)

sample_sentence[0] #seleciona a primeira sentença

#tokenização de palavras
list_words=[]
for i in range(len(sample_sentence)):
  sample_words = nltk.word_tokenize(text=sample_sentence[i])
  list_words.extend(sample_words)

print(list_words)  #corpus a ser analisado

#tokeniza palavras
def tokenizaPalavras(sentenca):
  sample_words = nltk.word_tokenize(text=sentenca)
  return sample_words

#removendo stopwords e criando o BoW
def removeStopWords(list_of_words):
  my_stop_words=['o','em','as','de','sao','nesta','.','e','a','na','do'] # cria a lista de stopwords
  list_cleaned=set(list_of_words)-set(my_stop_words)
  return list_cleaned

my_BoW=removeStopWords(list_words)

print(my_BoW)

#Cria o vetor que representa a sentenca na BoW 
def bagofwords(sentence, words):
    sentence_words = tokenizaPalavras(sentence)
    # conta a frequência de palavras que estão no vetor do BoW
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i,word in enumerate(words):
            if word == sw: 
                bag[i] += 1
    return sorted(zip(bag, words))

sentenca_teste='o igti oferece especializacao em deep learning e o igti oferece bootcamp'
print(bagofwords(sentenca_teste,my_BoW))