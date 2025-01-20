import gradio as gr
from sentiment_classifier import SentimentClassifier

classifier = SentimentClassifier()

def analyze_sentiment(text):
    result = classifier.predict(text)
    return f"Sentiment: {result['label']}\nConfidence: {result['score']}"

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter text to analyze", lines=3),
    outputs=gr.Textbox(label="Result"),
    title="Sentiment Analysis",
    description="Enter some text to analyze its sentiment (positive/negative)."
)

if __name__ == "__main__":
    demo.launch() 