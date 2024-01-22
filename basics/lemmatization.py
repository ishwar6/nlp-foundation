# lemmatization in NLTK can be more effective when you also provide the part of speech for each word. 

# Lemmatization: This process involves reducing words to their base or dictionary form (lemma). 
# Effectiveness with POS Tags: When you provide the part of speech for each word to the lemmatizer, it becomes more effective. 
# This is because many words have different lemmas based on their part of speech.
# For example, the word "set" can be a noun ("a set of books") or a verb ("to set a table"), and its lemma changes accordingly.

# Combining POS Tagging with Lemmatization: In advanced NLP tasks, it's common to first tag words with their POS and then lemmatize them using these tags.
# This approach ensures that the lemmatization process takes into account the context and the role of each word in a sentence, leading to more accurate results.


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize



# The averaged_perceptron_tagger in NLTK (Natural Language Toolkit) is a specific type of part-of-speech (POS) tagger. 
# It uses the averaged perceptron algorithm to assign POS tags to words in a text. 
  
nltk.download('averaged_perceptron_tagger')


# Function to convert NLTK's POS tags to WordNet's format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default case

lemmatizer = WordNetLemmatizer()

text = "Glad to have received the blessings"
tokens = word_tokenize(text)
tagged = pos_tag(tokens)

lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
print(lemmatized)
