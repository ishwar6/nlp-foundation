import nltk
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


# Stopwords
#  Commonly used words (such as "the", "a", "an", "in") that are often removed in the data preprocessing step.
filtered_words = [word for word in words if word not in stopwords.words('english')]
FilteredWords =  ['This', '22nd', 'Jan', '2023', ':', 'Happy', 'new', 'Shri', 'Ram', 'Mandir', 'Ayodhya', '.', 
                 'Its', 'astounding', 'architecture', 'set', 
                 'impress', 'anyone', 'pays', 'visit', '.', 'Glad', 'received', 'Shri', 'Ram', '’', 'blessings', '.']

# total words decreased to 28 from 41


# 3. Stemming and Lemmatization
# Stemming: Reducing words to their base or root form. It's a crude heuristic that chops off the ends of words.
# Lemmatization: Similar to stemming, but brings context to the words. It links words with similar meaning to one word.

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

Lemmatized_Words= ['This', '22nd', 'Jan', '2023', ':', 'Happy', 'new', 'Shri', 'Ram', 'Mandir', 'Ayodhya', '.', 
                   'Its', 'astounding', 'architecture', 'set', 'impress', 'anyone', 'pay', 'visit', '.', 
                   'Glad', 'received', 'Shri', 'Ram', '’', 'blessing', '.']


#Lemmatization is a process in NLP where words are reduced to their base or dictionary form (called lemma). 
# Unlike stemming, lemmatization considers the context and converts the word to its meaningful base form.

# Tagging
tagged = pos_tag(filtered_words)


