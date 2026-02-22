# 🎬 IMDB Sentiment Model

An LSTM-based sentiment classifier for IMDB movie reviews, built with TensorFlow/Keras. The app runs entirely on CPU with no GPU required.

## 📦 Technologies

- `Python`
- `TensorFlow / Keras`
- `Jupyter Notebook`
- `Streamlit`
- `NumPy`

## 🦄 Features

Here's what you can do with the Sentiment Model:

- **Train the Model**: Run the notebook end-to-end to download the IMDB dataset, train the LSTM, and save the model automatically.

- **Predict Sentiment**: Type any movie review into the Streamlit app and get an instant positive or negative prediction with a confidence score.

- **Low Resource**: The model runs comfortably on CPU with less than 4GB of RAM, no need for GPU or cloud setup.

- **Early Stopping**: Training uses `EarlyStopping` to avoid overfitting and stop at the best validation checkpoint automatically.

- **Visual Training Curves**: Accuracy and loss plots are saved after training so you can inspect how the model learned over each epoch.

## 👩🏽‍🍳 The Process

I started by exploring the IMDB dataset built into Keras, understanding how reviews are tokenized and padded to a fixed sequence length. From there, I built the model architecture with an Embedding layer, an LSTM layer, and a Dense output, keeping it small enough to train quickly on CPU.

Once training was working, I added `EarlyStopping` and experimented with hyperparameters like vocabulary size, embedding dimensions, and sequence length to find a balance between speed and accuracy.

After saving the trained model, I wrote a Streamlit app that loads it and lets anyone type a review and see a live prediction. I wired the notebook to write the app file automatically on the final cell so the whole workflow runs in one place.

## 📚 What I Learned

### 🧠 Recurrent Networks and Sequence Modeling:

- **Sequential Text**: Working with LSTMs taught me how a network can carry memory across tokens in a sequence, which is fundamentally different from feedforward architectures.
- **Vanishing Gradients**: I learned why plain RNNs struggle with long sequences and how the LSTM gating mechanism addresses that problem.

### 📏 Text Preprocessing:

- **Tokenization and Padding**: I had to understand how raw text is mapped to integer sequences and why padding to a fixed length is necessary for batched training.
- **Vocabulary Size Trade-offs**: Limiting the vocabulary to the top 10,000 words keeps memory manageable while still capturing the most meaningful tokens.

### 🎨 Embedding Layers:

- **Learned Representations**: I discovered how an Embedding layer maps each word index to a dense vector that the network learns during training, rather than using hand-crafted features.

### 🔍 Regularization Techniques:

- **EarlyStopping**: Implementing early stopping showed me how to monitor validation loss and restore the best weights automatically to prevent overfitting.

### 🎣 Model Persistence and Deployment:

- **Saving and Loading**: I learned how to save a Keras model to disk and reload it in a separate application, which is the foundation of any real deployment workflow.
- **Streamlit**: Building the inference UI with Streamlit showed me how quickly a data science model can be turned into something interactive and shareable.

### 📈 Overall Growth:

This project connected theory I had read about sequence models to a working implementation. Going from raw text to a live prediction in the browser made the full pipeline feel more accessible, and the low-resource constraint pushed me to think carefully about every architectural choice rather than just scaling up.

## 💭 How can it be improved?

- Add bidirectional LSTM layers to capture context from both directions.
- Experiment with pre-trained word embeddings like GloVe or Word2Vec.
- Add a confidence threshold to surface uncertain predictions.
- Support batch inference so multiple reviews can be evaluated at once.
- Add a proper training configuration file instead of hardcoded hyperparameters.
- Display a token-level attribution heatmap to show which words drove the prediction.

## 🚦 Running the Project

To run the project in your local environment, follow these steps:

1. Clone the repository to your local machine.
2. Run `pip install -r requirements.txt` in the project directory to install the required dependencies.
3. Open `imdb_sentiment_analysis.ipynb` and **Run All Cells** to train the model. This takes around 2–5 minutes on CPU and saves `imdb_lstm_model.keras` and `streamlit_app.py` automatically.
4. Run `streamlit run streamlit_app.py` to launch the app.
5. Open [http://localhost:8501](http://localhost:8501) (or the address shown in your console) in your web browser to view the app.
