from nltk.corpus import brown, reuters, gutenberg
import nltk
from collections import Counter

# Download the necessary corpora
# nltk.download('brown')
# nltk.download('reuters')
# nltk.download('gutenberg')


def word_frequency(words):
    # Ensure that words is a list, even if a single word is passed
    if isinstance(words, str):
        words = [words]

    # Convert all input words to lowercase
    words_lower = [word.lower() for word in words]

    # Brown Corpus
    brown_words = [w.lower() for w in brown.words()]
    brown_counts = Counter(brown_words)

    # Reuters Corpus
    reuters_words = [w.lower() for w in reuters.words()]
    reuters_counts = Counter(reuters_words)

    # Gutenberg Corpus
    gutenberg_words = [w.lower() for w in gutenberg.words()]
    gutenberg_counts = Counter(gutenberg_words)

    # Initialize total frequency
    total_frequency = 0

    # Calculate total frequency across all corpora for all words
    for word in words_lower:
        total_frequency += brown_counts[word] + \
            reuters_counts[word] + gutenberg_counts[word]

    return total_frequency


# # Example usage
# # You can pass a list of words or a single word
# words_to_search = ['government', 'Iran']
# total_frequency = word_frequency(words_to_search)

# # Print the total frequency
# print(
#     f"The total frequency of the words {words_to_search} across all corpora is {total_frequency}.")
