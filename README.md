# Sentiment Analysis Classifier

A robust sentiment analysis tool that uses state-of-the-art transformer models to analyze the emotional tone of text. Built with PyTorch and Hugging Face's transformers library, featuring a user-friendly Gradio interface.

## Features

- 🎯 **Accurate Sentiment Analysis**: Uses DistilBERT model fine-tuned on SST-2 dataset
- 🌡️ **Confidence Scoring**: Provides confidence scores for predictions
- 🖥️ **Web Interface**: Easy-to-use Gradio UI for interactive analysis
- 🔄 **Automated Testing**: GitHub Actions for continuous testing
- 📊 **Inference Monitoring**: Automated sentiment analysis on diverse test cases

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Godnight1006/classifier.git
cd classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Make sure you have PyTorch installed. If not, install it following the instructions at [pytorch.org](https://pytorch.org/).

## Usage

### Web Interface

Run the Gradio web interface:
```bash
python app.py
```
Then open your browser and navigate to the URL shown in the console (typically http://127.0.0.1:7860).

### Python API

```python
from sentiment_classifier import SentimentClassifier

# Initialize the classifier
classifier = SentimentClassifier()

# Analyze a single text
result = classifier.predict("This is a great example!")
print(f"Sentiment: {result['label']}")
print(f"Confidence: {result['score']}")

# Quick positive/negative check
is_positive = classifier.is_positive("This is wonderful!")
```

## Model Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Fine-tuning**: SST-2 (Stanford Sentiment Treebank)
- **Output**: Binary classification (POSITIVE/NEGATIVE) with confidence scores
- **Input**: Any English text (recommended: 1-128 tokens)

## Continuous Integration

The project includes two GitHub Actions workflows:

1. **Test Workflow** (`test.yml`)
   - Runs unit tests on every push
   - Verifies core functionality
   - Tests both positive and negative cases

2. **Inference Workflow** (`inference.yml`)
   - Runs inference on diverse test cases
   - Tests handling of:
     - Clear positive/negative statements
     - Mixed or complex emotions
     - Subtle or neutral statements
   - Monitors model behavior over time

## Example Results

The model can handle various types of inputs:

```
Text: "This is a fantastic day!"
Sentiment: POSITIVE
Confidence: 0.9876

Text: "The experience was bittersweet."
Sentiment: NEGATIVE
Confidence: 0.5432

Text: "The weather is changing."
Sentiment: NEUTRAL
Confidence: 0.5123
```

## Testing

Run the test suite:
```bash
pytest
```

The tests verify:
- Positive sentiment detection
- Negative sentiment detection
- Binary classification accuracy
- Confidence score validity

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built using [Hugging Face Transformers](https://huggingface.co/transformers/)
- Web interface powered by [Gradio](https://gradio.app/)
- Model trained on [SST-2 dataset](https://nlp.stanford.edu/sentiment/)