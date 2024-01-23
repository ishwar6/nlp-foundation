# This is a technique to obtain vector representation of word. By a team lead by Tomáš Mikolov. 
# As the name implies, word2vec represents each distinct word with a particular list of numbers called a VECTOR. 

# https://en.wikipedia.org/wiki/Word2vec
######################################## What is WORD EMBEDDING? ################################################

# In natural language processing (NLP), a word embedding is a representation of a word. 
# The embedding is used in text analysis.

# Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way
# that the words that are closer in the vector space are expected to be similar in meaning.


######################################### Key Concepts of Word2Vec ################################################

1. It stores Contextual information: It learns representation of words in context of in which they appear. Words with similar context, tends to have similar embeddings. 
2. Based on CBOW (continuous Beg of word) and Skip-Gram models. 
3. Dimensionality and Dense vector. 
4. Training Process. 
