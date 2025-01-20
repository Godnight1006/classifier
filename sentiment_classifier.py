from transformers import pipeline

class SentimentClassifier:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")
    
    def predict(self, text):
        result = self.classifier(text)[0]
        return {
            'label': result['label'],
            'score': round(result['score'], 4)
        }
    
    def is_positive(self, text):
        result = self.predict(text)
        return result['label'] == 'POSITIVE' 