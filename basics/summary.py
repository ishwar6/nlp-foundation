import nltk

# Lets download few packages: 

# "Punkt" is a part of the Natural Language Toolkit (NLTK) library in Python, specifically designed for sentence tokenization. 
# Punkt works by using an unsupervised algorithm to learn abbreviations and other word types which commonly occur at the start of sentences. 

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# Example text
text = "This is 22nd Jan 2023: Happy to be at the new Shri Ram Mandir in Ayodhya. Its astounding architecture is set to impress anyone who pays a visit. Glad to have received Shri Ram’s blessings."

########################################## STEP - 1 ######################################################
# Tokenization
# Definition: The process of breaking down text into smaller units, such as words or sentences.
# Types:
# Word Tokenization: Splitting text into individual words.
# Sentence Tokenization: Dividing text into individual sentences.

words = word_tokenize(text)

Words= ['This', 'is', '22nd', 'Jan', '2023', ':', 
        'Happy', 'to', 'be', 'at', 'the', 'new', 'Shri', 'Ram', 'Mandir', 'in', 'Ayodhya', '.', 
        'Its', 'astounding', 'architecture', 'is', 'set', 'to', 'impress', 'anyone', 'who', 'pays', 'a', 'visit', '.', 
        'Glad', 'to', 'have', 'received', 'Shri', 'Ram', '’', 's', 'blessings', '.']
# total words: 41

sentences = sent_tokenize(text)
# Sentence Tokenization: Dividing text into individual sentences.
Sentences = ['This is 22nd Jan 2023: Happy to be at the new Shri Ram Mandir in Ayodhya.', 
            'Its astounding architecture is set to impress anyone who pays a visit.', 'Glad to have received Shri Ram’s blessings.']


########################################## STEP - 2 ######################################################
# Stopwords
#  Commonly used words (such as "the", "a", "an", "in") that are often removed in the data preprocessing step.
filtered_words = [word for word in words if word not in stopwords.words('english')]
FilteredWords =  ['This', '22nd', 'Jan', '2023', ':', 'Happy', 'new', 'Shri', 'Ram', 'Mandir', 'Ayodhya', '.', 
                 'Its', 'astounding', 'architecture', 'set', 
                 'impress', 'anyone', 'pays', 'visit', '.', 'Glad', 'received', 'Shri', 'Ram', '’', 'blessings', '.']

# total words decreased to 28 from 41


########################################## STEP - 3 ######################################################
# 3. Stemming and Lemmatization

stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]


STEMS = ['thi', '22nd', 'jan', '2023', ':', 'happi', 'new', 'shri', 'ram', 'mandir', 'ayodhya', '.', 
         'it', 'astound', 'architectur', 'set', 'impress', 'anyon', 'pay', 'visit', '.', 'glad', 'receiv', 'shri', 'ram', '’', 'bless', '.']


# Stemming: Reducing words to their base or root form. It's a crude heuristic that chops off the ends of words.
# When Not to Use Stemming
# In tasks requiring high accuracy and context understanding, like sentiment analysis. Stemming might oversimplify and lose context.
# When the application involves understanding the meaning of the word accurately (e.g., language translation).



# Lemmatization: Similar to stemming, but brings context to the words. It links words with similar meaning to one word.

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

Lemmatized_Words= ['This', '22nd', 'Jan', '2023', ':', 'Happy', 'new', 'Shri', 'Ram', 'Mandir', 'Ayodhya', '.', 
                   'Its', 'astounding', 'architecture', 'set', 'impress', 'anyone', 'pay', 'visit', '.', 
                   'Glad', 'received', 'Shri', 'Ram', '’', 'blessing', '.']


#Lemmatization is a process in NLP where words are reduced to their base or dictionary form (called lemma). 
# Unlike stemming, lemmatization considers the context and converts the word to its meaningful base form.


########################################## STEP - 4 ######################################################
# Tagging
# It's the process of labeling each word in a sentence with its appropriate part of speech, such as noun, verb, adjective, etc.
tagged = pos_tag(filtered_words)


Tagged= [('This', 'DT'), ('22nd', 'CD'), ('Jan', 'NNP'), ('2023', 'CD'), (':', ':'), ('Happy', 'JJ'), ('new', 'JJ'),
          ('Shri', 'NNP'), ('Ram', 'NNP'), ('Mandir', 'NNP'), ('Ayodhya', 'NNP'), ('.', '.'),
            ('Its', 'PRP$'), ('astounding', 'JJ'), ('architecture', 'NN'), ('set', 'VBN'), ('impress', 'JJ'), ('anyone', 'NN'),
              ('pays', 'VBZ'), ('visit', 'NN'), ('.', '.'), 
         ('Glad', 'NNP'), ('received', 'VBD'), ('Shri', 'NNP'), ('Ram', 'NNP'), ('’', 'NNP'), ('blessings', 'NNS'), ('.', '.')]

# 'DT': Determiner. E.g., "This"
# 'CD': Cardinal number. E.g., "22nd", "2023"
# 'NNP': Proper noun, singular. E.g., "Jan", "Shri", "Ram", "Mandir", "Ayodhya", "Glad"
# ':': Punctuation mark, colon or ellipsis.
# 'JJ': Adjective. E.g., "Happy", "new", "astounding", "impress"
# 'NN': Noun, singular or mass. E.g., "architecture", "anyone", "visit"
# 'VBN': Verb, past participle. E.g., "set"
# 'VBZ': Verb, 3rd person singular present. E.g., "pays"
# 'VBD': Verb, past tense. E.g., "received"
# 'NNS': Noun, plural. E.g., "blessings"
# '.': Punctuation mark, sentence closer.
# 'PRP$': Possessive pronoun. E.g., "Its"
# '’': Punctuation mark (apostrophe).

