from nltk.corpus import brown, reuters, gutenberg
import nltk
from collections import Counter

# Download the necessary corpora
nltk.download('brown')
nltk.download('reuters')
nltk.download('gutenberg')


def word_frequency(word):
    # Convert the input word to lowercase
    word_lower = word.lower()

    # Brown Corpus
    brown_words = [w.lower() for w in brown.words()]
    brown_counts = Counter(brown_words)
    brown_frequency = brown_counts[word_lower]

    # Reuters Corpus
    reuters_words = [w.lower() for w in reuters.words()]
    reuters_counts = Counter(reuters_words)
    reuters_frequency = reuters_counts[word_lower]

    # Gutenberg Corpus
    gutenberg_words = [w.lower() for w in gutenberg.words()]
    gutenberg_counts = Counter(gutenberg_words)
    gutenberg_frequency = gutenberg_counts[word_lower]

    # Sum of frequencies in all corpora
    total_frequency = brown_frequency + reuters_frequency + gutenberg_frequency

    # Return individual and total frequencies
    return {
        'Brown Corpus': brown_frequency,
        'Reuters Corpus': reuters_frequency,
        'Gutenberg Corpus': gutenberg_frequency,
        'Total': total_frequency
    }


# Example usage
word_to_search = 'test'
frequencies = word_frequency_in_corpora(word_to_search)

# Print the frequencies
for corpus, freq in frequencies.items():
    print(f"The word '{word_to_search}' appears {freq} times in {corpus}.")
