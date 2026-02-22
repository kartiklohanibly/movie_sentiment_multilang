import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data safely
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Cleans input text by:
    - Lowercasing
    - Removing special characters
    - Tokenizing
    - Removing stopwords
    """

    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return " ".join(filtered_tokens)
