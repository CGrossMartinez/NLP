import glob
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer


document_path = "C:/Users/Carlos/Desktop/MS Courses/CAP6776 Information Retrieval/Assignment1/Articles"

article_list = glob.glob(document_path + '/*.txt')

articles = []

for i in range (0, len(article_list)):
    temp_string = ''

    content = open(article_list[i], 'r').readlines()

    for sentence in content:
        temp_string += sentence

    articles.append(temp_string)


exclude = set(string.punctuation)
temp_article_0 = articles[0]
temp_article_1 = articles[1]
temp_article_2 = articles[2]


temp_article_0_no_punctuation = ''.join(ch for ch in temp_article_0 if ch not in exclude)
temp_article_1_no_punctuation = ''.join(ch for ch in temp_article_1 if ch not in exclude)
temp_article_2_no_punctuation = ''.join(ch for ch in temp_article_2 if ch not in exclude)

#temp_article_0_no_punctuation = temp_article_0_no_punctuation.toarray()

print(type(temp_article_0_no_punctuation))
#print(temp_article_1_no_punctuation)
#print(temp_article_2_no_punctuation)



tokenized_article_0 = word_tokenize(temp_article_0_no_punctuation)
tokenized_article_1 = word_tokenize(temp_article_1_no_punctuation)
tokenized_article_2 = word_tokenize(temp_article_2_no_punctuation)

#print(tokenized_article_0)
#print(tokenized_article_1)
#print(tokenized_article_2)

stop_words = set(stopwords.words("english"))

#print(stop_words)

filtered_article_0 = []
filtered_article_1 = []
filtered_article_2 = []

for word in tokenized_article_0:
    if word.lower() not in stop_words:
        filtered_article_0.append(word)

for word in tokenized_article_1:
    if word.lower() not in stop_words:
        filtered_article_1.append(word)

for word in tokenized_article_2:
    if word.lower() not in stop_words:
        filtered_article_2.append(word)

#print(filtered_article_0)
#print(filtered_article_1)
#print(filtered_article_2)

stemmed_article_0 = []
stemmed_article_1 = []
stemmed_article_2 = []

ps = PorterStemmer()

for x in range (0, len(filtered_article_0)):
    stemmed_article_0.append(ps.stem(filtered_article_0[x]))

for x in range (0, len(filtered_article_1)):
    stemmed_article_1.append(ps.stem(filtered_article_1[x]))

for x in range (0, len(filtered_article_2)):
    stemmed_article_2.append(ps.stem(filtered_article_2[x]))

#print(stemmed_article_0)
#print(stemmed_article_1)
#print(stemmed_article_2)

clean_articles = []

clean_articles.append(stemmed_article_0)
clean_articles.append(stemmed_article_1)
clean_articles.append(stemmed_article_2)

#print(clean_articles[0])
#print(clean_articles[1])
#print(clean_articles[2])


tfidf = TfidfVectorizer(stemmed_article_0, stop_words='english')

tfs = tfidf.fit_transform(temp_article_0_no_punctuation)

#print(tfidf.vocabulary_)

doc_matrix = tfs.toarray()

print(doc_matrix)



#for sentence, feature in zip(temp_article_0_no_punctuation, tfidf_features):
    #print(sentence)
    #print(feature)

#print(tfs)










    




        
        
   


        


  














