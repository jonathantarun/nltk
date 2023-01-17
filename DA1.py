#Jonathan Tarun - 20BCE1778
#NLP_DA1

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from urllib import request
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

text = brown.words()
para = ' '.join([str(elem) for elem in text])

brown.words(categories='government')
print(para)

#1.1 - Explore Brown Corpus and find the size, tokens, categories
print("Explore Brown Corpus and find the size, tokens, categories")
print(brown.categories())
#1.2 - Find the size of word tokens
print("Find the size of word tokens")
print(len(word_tokenize(para)))
#1.3 - Find the size of the category “government” 
print("Find the size of the category “government” ") 
print(len(brown.words(categories='government')))
#1.4 - List the most frequent tokens 
dict = defaultdict(int)
for sub in para:
    for wrd in sub.split():
        dict[wrd] += 1
freq = max(dict, key=dict.get)
print("Word with maximum frequency : " + str(freq))
#1.5 - Count the number of sentences 
print("Count the number of sentences") 
no_of_sentences = sent_tokenize(para)
print(len(no_of_sentences))

#2.1
#Raw Corpus

#To answer these Questions:
#->How can we create software that can access text from both local files and the internet?
#->How can we split documents up into individual words and punctuation symbols?
#->How can we write programs to produce formatted output and save it in a file?

#2.1.1 - Accessing Text from the Web 
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
res = request.urlopen(url)
rawtext = res.read().decode('utf8')
type(rawtext)   
len(rawtext)
rawtext[:75]

#2.1.2 Strings: Text Processing at the Lowest Level

#Accessing Individual Characters
print(text[58])

#Accessing Substrings
print(text[-19:-2])
print(text[50:100])
#More Queries
lorem = """This is our first digital assignment in natural language processing done by Jonathan Tarun - 20BCE1778."""
 
# upper() function to convert
# string to upper case
print("\nConverted String:")
print(lorem.upper())
 
# lower() function to convert
# string to lower case
print("\nConverted String:")
print(lorem.lower())
 
# converts the first character to
# upper case and rest to lower case
print("\nConverted String:")
print(lorem.title())
 
# original string never changes
print("\nOriginal String")
print(lorem)

#2.1.3 Extracting encoded text from files
path = nltk.data.find('C:\Users\Emmanuel\Downloads\enctext.txt') #Emmanuel is my father's name
f = open(path, encoding='latin2')
for line in f:
    line = line.strip()
    print(line.encode('unicode_escape'))

#2.2 POS Tagging
#POS Tagging (Parts of Speech Tagging) is a process to mark up the words in text format for a particular part of a speech based on its definition and context. It is responsible for text reading in a language and assigning some specific token (Parts of Speech) to each word. It is also called grammatical tagging.
#COUNTING POS TAGS
lower_case = para.lower()
tokens = nltk.word_tokenize(lower_case)
tags = nltk.pos_tag(tokens)
counts = Counter( tag for word,  tag in tags)
print("COUNTING POS TAGS: number of tags=")
print(counts)
#Tagging Sentences
print("Tagging Sentences: ")
sen = nltk.sent_tokenize(para)
for sent in sen:
     print(nltk.pos_tag(nltk.word_tokenize(sent)))
#3.1  Word segmentation
#Breaking down a string of written language into its individual words is known as word segmentation.
sentences = nltk.sent_tokenize(para)

for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    print(words)
    print()

#3.2 Sentence Tokenization
#Breaking down a string of written language into its individual phrases is known as sentence segmentation. 
#When we encounter a punctuation mark, we may break apart the phrases in English.

for sentence in sentences:
    print(sentence)
    print()

#3.3 Convert to Lowercase
text = [token.lower() for token in text]
print(text)

#3.4 Stop words removal
stop_words = set(stopwords.words('english'))
  
word_tokens = word_tokenize(para)

#converts the words in word_tokens to lower case and then checks whether they are present in stop_words or not
filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]

#with no lower case conversion
filtered_sentence = []
  
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)
  
print(word_tokens)
print(filtered_sentence)

# 3.5 Stemming
#Stemming is the process of reducing a word to its stem that affixes to suffixes and prefixes or to the roots of words known as "lemmas".
ps = PorterStemmer()
words = ["consultant", "consulting", "consults", "consulting", "consultative"]
  
for w in words:
    print(w, " : ", ps.stem(w)) 

#3.6 Lemmatization
#Lemmatization is a text normalization technique used in Natural Language Processing (NLP), that switches any kind of a word to its base root mode.
lm = WordNetLemmatizer()
  
print("rocks :", lm.lemmatize("trees"))
print("corpora :", lm.lemmatize("children"))
  
# a denotes adjective in "pos"
print("better :", lm.lemmatize("better", pos ="a"))

#3.7 Part of speech tagger
#A POS tagger's main objective is to assign linguistic information, primarily grammatical information, to sub-sentential units.
#These units are referred to as tokens, and they typically correspond to words and symbols (e.g. punctuation).

stop_words = set(stopwords.words('english'))
tokenized = sent_tokenize(text)
for i in tokenized:
     
    # Word tokenizers is used to find the words and punctuation in a string
    wordsList = nltk.word_tokenize(i)
 
    # removing stop words from wordList
    wordsList = [w for w in wordsList if not w in stop_words]
 
    #  Using a POS-tagger.
    tagged = nltk.pos_tag(wordsList)
 
    print(tagged)
