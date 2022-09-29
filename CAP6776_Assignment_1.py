#Carlos Gross-Martinez
#CAP6776-Information Retrieval
#Assignment 1 - NLTK

#importing libraries
import os 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import  TfidfTransformer

#variable declaratrion and initialization
doc_names = []
token_dict = {}
stemmed_words = [] 
no_stop_words = []
all_stemmed_words = []

#Creating variable to add spaces when saving results in text file
text_lines = "\n"

#deifining varible with stop words and punctuation for comparison when cleaning text files
stop_words = set(stopwords.words("english")) | set(string.punctuation) 

#saving path of files location into variable "data_path"
data_path = 'C:/Users/Carlos/Desktop/MS Courses/CAP6776 Information Retrieval/Assignment1/Articles'

#importing porter stemmer model
ps = PorterStemmer()

#importing TF-IDF vectorizer model 
tfidf = TfidfVectorizer(stop_words='english')

#declaring path to save out file with results
output_file = open('C:/Users/Carlos/Desktop/MS Courses/CAP6776 Information Retrieval/Assignment1/dataoutput.txt', 'w')

#loop to get to location of articles
for subdir, dirs, files in os.walk(data_path):
    #loop that checks all files in path
    for file in files:
        #updating path to add file name information to it
        file_path = subdir + os.path.sep + file
        #opening file in read mode
        file_contents = open(file_path, 'r')
        #condition that checks to ensure file opened is a text file
        if '.txt' in file_path:
            #saving contents of text file in variable "text"
            text = file_contents.read()
            #setting all text to lower case in file
            lowered = text.lower()
            #saving file name and information to dictionary
            token_dict[file] = lowered
            #closing file
            file_contents.close()

# saving the number of articles into variable
num_docs = len(token_dict)

#print(num_docs)
#print(token_dict)

#loop that traverses through all text file names in dictionary
for file_name in token_dict.keys():
    #saving names of files into variable doc_names
    doc_names.append(file_name)

#print(doc_names)

#Data cleaning and tokenization

#loope that traverses through all file names in dictionary
for file in token_dict.keys():

    #tokenizing sentences and words in articles

    #tokenizing content in file names and saving it to varibale "words"
    words = word_tokenize(token_dict[file]) 

    #tokenizing senteces in files and saving informaiton to output_file
    output_file.write((text_lines + "Sentence Tokenizing" + ' ' + file + " .\n" + text_lines))
    output_file.write(str(sent_tokenize(token_dict[file])) + "\n")

    #tokenizing senteces in files and saving informaiton to output_file
    output_file.write((text_lines + "Word Tokenizing" + ' ' + file + " .\n" + text_lines))
    output_file.write(str(words) + "\n")

    #Removing stop words
    
    #loops that traverses through all tokenized words
    for w in words:
        #condition that checks if tokenize word is a stop word
        if w not in stop_words:
            #saving tokenized word to variable if it is not a stop word
            no_stop_words.append(w)
    
    #saving information of tokenized words with no stop words into output_file
    output_file.write((text_lines + "Stop Words Removed From:" + ' ' + file + " .\n" + text_lines))
    output_file.write(str(no_stop_words) + "\n")

    #Stemming tokenized words after removing stop words

    #loops that traverses through all tokenized words
    for w in words:
        #condition that checks if tokenize word is a stop word
        if w not in stop_words:
            #stemming word and saving it into variable
            stemmed_words.append(ps.stem(w))

    #saving information of stemmed words with no stop words into output_file
    output_file.write((text_lines + "Stemming" + ' ' + file + " .\n" + text_lines))
    output_file.write(str(stemmed_words) + "\n")

    #saving the cleand data into variable
    all_stemmed_words.append(stemmed_words)

#TF-IDF Calculation

#conducting TF-IDF calculations by fitting and transforming the content of the articles
tfs = tfidf.fit_transform(token_dict.values())
#converting results into and array and saving to variable
doc_matrix = tfs.toarray()
#saving all features (words) from all documents into variables
set_vocab = tfidf.get_feature_names()

#saving information TS-IDF Calculations into output_file
output_file.write(text_lines + "TD-IDF Document-Word Matrix. \n" + text_lines)
output_file.write("%-15s%-20s%-20s%-20s\n" % ("Words", str(doc_names[0]), str(doc_names[1]), str(doc_names[2])))

#loop tranverse though all features(words) and provides TF-IDF value per document examined
for i in range (0, len(set_vocab)):
    output_file.write("%-15s%-20s%-20s%-20s\n" % (str(set_vocab[i]), str(doc_matrix[0][i]), str(doc_matrix[1][i]), str(doc_matrix[2][i])))

# Consine Similarity calculation

#saving header information for cosine similarity in outpu_file
output_file.write(text_lines + "Cosine Similarity. \n" + text_lines)

#loop that traverses through all ducuments for cosine similarity comparison
for i in range(0, num_docs):
    #loop that traverses through all ducuments for cosine similarity comparison
    for j in range(i, num_docs):
        #conditionthat checks that a file is not compared to itself
        if i != j:
            #saving to output_file results of consine similarity calculations
            output_file.write("Cosine similarity of %s to %s is %s. \n" % (doc_names[i], doc_names[j], cosine_similarity(tfs[i,], tfs[j,])))

#closing output_file
output_file.close()