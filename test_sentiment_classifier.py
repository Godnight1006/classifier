import pytest
from sentiment_classifier import SentimentClassifier

@pytest.fixture
def classifier():
    return SentimentClassifier()

def test_positive_sentiment(classifier):
    text = "I love this product! It's amazing!"
    result = classifier.predict(text)
    assert result['label'] == 'POSITIVE'
    assert result['score'] > 0.5

def test_negative_sentiment(classifier):
    text = "This is terrible, I hate it."
    result = classifier.predict(text)
    assert result['label'] == 'NEGATIVE'
    assert result['score'] > 0.5

def test_is_positive(classifier):
    assert classifier.is_positive("This is wonderful!") == True
    assert classifier.is_positive("This is horrible!") == False 