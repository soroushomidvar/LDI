import nltk
# nltk.download('punkt', force=True)
# nltk.download('brown')
# nltk.download('reuters')
# nltk.download('gutenberg')
# nltk.download('punkt_tab')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def compute_bleu(reference: str, candidate: str) -> float:
    """
    Compute BLEU-1 score (unigram precision) between a reference text and a candidate text.
    
    :param reference: The ground truth text.
    :param candidate: The generated text.
    :return: BLEU-1 score (between 0 and 1).
    """
    smoothing = SmoothingFunction().method1
    reference_tokens = [nltk.word_tokenize(reference)]  # BLEU expects a list of reference lists
    candidate_tokens = nltk.word_tokenize(candidate)
    return sentence_bleu(reference_tokens, candidate_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoothing)  # Unigram precision only

def compute_rouge(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 (unigram) precision, recall, and F1-score between a reference and candidate text.
    
    :param reference: The ground truth text.
    :param candidate: The generated text.
    :return: A dictionary containing ROUGE-1 precision, recall, and F1-score.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, candidate)['rouge1']
    
    return {
        "P": scores.precision,
        "R": scores.recall,
        "F1": scores.fmeasure
    }


# Example usage
# reference_text = "sony"
# candidate_text = "sony inc"

# print("BLEU Score:", compute_bleu(reference_text, candidate_text))
# print("ROUGE Scores:", compute_rouge(reference_text, candidate_text))
