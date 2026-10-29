# 📰 AG News Classification Using Transformer Encoder, BiLSTM, and DistilBERT

## 📌 Project Overview

News articles are published every second across the world. Manually organizing these articles into categories can be time-consuming. **Natural Language Processing (NLP)** enables computers to understand and classify text automatically.

This project implements and compares three deep-learning models for **news article classification** using the AG News dataset:

1. **Transformer Encoder** — built and trained from scratch.
2. **BiLSTM (Bidirectional Long Short-Term Memory)** — built and trained from scratch.
3. **DistilBERT** — a pretrained Transformer model fine-tuned for news classification.

The objective is to understand how these models learn from text, evaluate their performance, and identify the advantages and limitations of each approach.

### 🎯 Project Objectives

- Understand how machine learning models process and classify text.
- Implement a **Transformer Encoder and BiLSTM from scratch**.
- Fine-tune a pretrained **DistilBERT** model.
- Compare the models using accuracy, Macro F1-score, and other evaluation metrics.
- Analyze classification performance using confusion matrices, comparison tables, and visualizations.
- Understand the trade-offs between training from scratch and using pretrained language models.

---

## 📂 Dataset: AG News

The **AG News dataset** is a widely used benchmark for news-topic classification. It contains news articles labeled according to their subject.

The commonly used version contains **120,000 training samples and 7,600 test samples**, divided into four categories.

| Label | Category | Description |
|---|---|---|
| 0 | **World** | International news, politics, conflicts, and global events |
| 1 | **Sports** | Sports teams, matches, tournaments, and athletes |
| 2 | **Business** | Companies, financial markets, trade, and the economy |
| 3 | **Sci/Tech** | Science, technology, research, and computing |

### Example

**Input news article:**

> The football team secured a victory after scoring two goals in the final minutes of the match.

**Expected prediction:** `Sports`

The model processes the article and predicts the category it considers most appropriate.

---

## 🧠 Fundamental Concepts

Before understanding the three models, it helps to know a few basic NLP concepts.

### 1. Natural Language Processing (NLP)

**Natural Language Processing** is a branch of Artificial Intelligence that enables computers to process and analyze human language.

Examples include:
- Spam email detection
- Sentiment analysis
- Language translation
- Chatbots
- News classification

### 2. Tokenization

Computers cannot directly process sentences the way humans do. **Tokenization** divides text into smaller units called *tokens*.

For example:

`"Technology is growing rapidly."`

A simple tokenizer might produce:

`["Technology", "is", "growing", "rapidly", "."]`

The exact tokens depend on the tokenizer used by the model.

### 3. Vocabulary and Token IDs

A **vocabulary** is a collection of tokens mapped to numerical IDs.

For example:

| Token | Example ID |
|---|---:|
| technology | 25 |
| business | 41 |
| sports | 67 |

These IDs are illustrative. The actual IDs depend on the vocabulary and tokenizer.

### 4. Word Embeddings

An **embedding** converts a token ID into a vector of numbers that a neural network can learn from.

Instead of working directly with words, the model learns numerical representations that help it recognize useful patterns in text.

### 5. Training and Loss

During training, a model predicts a category and compares its prediction with the correct label.

A **loss function** measures how far the prediction is from the correct answer. An optimizer updates the model's parameters to reduce this loss.

An **epoch** represents one complete pass through the training dataset.

### 6. Classification Layer

The **classification layer** converts the learned text representation into scores for the four categories. The category with the highest score becomes the model's predicted label.

---

## 🚀 Model 1: Transformer Encoder (Built from Scratch)

### What Is a Transformer?

A **Transformer** is a neural-network architecture that uses a mechanism called **attention** to learn relationships between different parts of a sequence.

For example, consider:

> The company released a new product because it wanted to attract more customers.

To understand the word *it*, the model may need to connect it with *the company*. Attention allows the model to use information from other tokens in the sentence.

### What Is Self-Attention?

**Self-attention** allows each token to determine how much information it should gather from other tokens in the same sequence.

At a high level, attention uses three representations:

- **Query:** what information a token is looking for.
- **Key:** what information a token can offer for matching.
- **Value:** the information that is combined using the attention scores.

The attention mechanism calculates relationships between tokens and uses them to construct context-aware representations.

### Architecture

The custom Transformer Encoder consists of:

1. **Embedding Layer:** converts token IDs into numerical vectors.
2. **Positional Encoding:** provides information about token positions.
3. **Transformer Encoder Layers:** learn contextual relationships using self-attention and feed-forward networks.
4. **Pooling Layer:** combines token representations into an article representation.
5. **Classification Layer:** predicts one of the four news categories.

### What Does "Built from Scratch" Mean?

In this project, the Transformer Encoder's model weights are **randomly initialized** and learned during training on AG News. The model does not start from pretrained language-model weights.

PyTorch is used to implement the architecture and training process.

### Advantages

- Learns relationships between different words, including distant words.
- Can process multiple token positions in parallel during training.
- Provides a strong attention-based architecture for text classification.

### Limitations

- Must learn useful language representations from the task's training data.
- Attention can require substantial memory for longer sequences.
- Performance depends on architecture, hyperparameters, and training duration.

---

## 🔄 Model 2: BiLSTM (Built from Scratch)

### What Is an LSTM?

**LSTM** stands for **Long Short-Term Memory**. It is a type of Recurrent Neural Network (RNN) designed to process sequential information.

An LSTM uses gates to control what information should be retained, updated, or forgotten as it processes a sequence.

For text classification, this helps the model learn how words relate to the words that appear before or after them.

### What Does Bidirectional Mean?

A normal LSTM reads a sequence in one direction. A **Bidirectional LSTM (BiLSTM)** contains two LSTM networks:

- **Forward LSTM:** reads the text from beginning to end.
- **Backward LSTM:** reads the text from end to beginning.

Their representations are combined so that the model can use context from both directions.

For example, the meaning of a word may become clearer when the model considers both the preceding and following words.

### Architecture

The BiLSTM consists of:

1. **Embedding Layer:** converts token IDs into vectors.
2. **Forward LSTM:** processes the sequence from left to right.
3. **Backward LSTM:** processes the sequence from right to left.
4. **Pooling Layer:** summarizes the sequence representations.
5. **Classification Layer:** predicts the news category.

### What Does "Built from Scratch" Mean?

The BiLSTM starts with **randomly initialized weights** and learns from the AG News training dataset rather than using a pretrained language model.

### Advantages

- Learns sequential patterns in text.
- Uses context from both directions.
- Provides a useful comparison with attention-based models.

### Limitations

- Sequential processing can make training slower.
- Very long sequences can be challenging to process efficiently.
- The model must learn language representations from the available training data.

---

## 🤖 Model 3: DistilBERT (Pretrained Transformer)

### What Is BERT?

**BERT** stands for **Bidirectional Encoder Representations from Transformers**. It is a Transformer-based language model pretrained on large collections of text.

During pretraining, BERT learns general language patterns that can later be adapted to specific tasks.

### What Is DistilBERT?

**DistilBERT** is a smaller, distilled version of BERT. It was designed to retain much of BERT's language-processing ability while reducing model size and computational requirements.

Unlike the custom Transformer Encoder and BiLSTM, DistilBERT begins with **pretrained model weights**.

### What Is Fine-Tuning?

**Fine-tuning** means taking a model that has already learned general language representations and training it further on a particular task.

In this project:

1. The DistilBERT tokenizer converts news text into model-specific token IDs.
2. The pretrained DistilBERT model processes the input.
3. A classification layer produces scores for the four news categories.
4. The model is trained on labeled AG News examples to adapt its predictions to news classification.

### Advantages

- Benefits from language representations learned during pretraining.
- Often performs strongly on text-classification tasks.
- Uses a tokenizer designed for the pretrained model.

### Limitations

- Requires downloading pretrained weights and tokenizer files.
- Fine-tuning can require significant computing resources.
- Performance depends on the pretrained checkpoint and fine-tuning settings.

---

## ⚖️ Comparison of the Three Models

| Feature | Transformer Encoder | BiLSTM | DistilBERT |
|---|---|---|---|
| Architecture | Transformer Encoder | Bidirectional LSTM | Distilled Transformer |
| Initial weights | Randomly initialized | Randomly initialized | Pretrained |
| Training approach | Trained from scratch | Trained from scratch | Fine-tuned |
| Context mechanism | Self-attention | Forward and backward recurrence | Self-attention |
| Tokenization | Project vocabulary | Project vocabulary | DistilBERT tokenizer |
| Pretrained language knowledge | No | No | Yes |

**Important:** This comparison evaluates both architecture and training approach. DistilBERT benefits from pretraining, whereas the custom Transformer Encoder and BiLSTM learn from AG News starting with randomly initialized weights.

---

## ⚙️ Project Workflow

The project follows these major steps:

1. **Dataset Loading:** load AG News and its category labels.
2. **Text Preprocessing:** tokenize the text and convert tokens into numerical representations.
3. **Model Construction:** initialize the custom Transformer Encoder and BiLSTM.
4. **Pretrained Model Loading:** load DistilBERT and its tokenizer.
5. **Model Training:** train the custom models and fine-tune DistilBERT.
6. **Evaluation:** generate predictions for the held-out test data.
7. **Metric Calculation:** calculate accuracy, Macro F1-score, and class-wise metrics.
8. **Visualization:** create comparison charts, confusion matrices, and training curves.
9. **Output Saving:** save model weights and result tables when the corresponding notebook cells are executed.

---

## 📊 Evaluation Metrics

### 1. Accuracy

**Accuracy** measures the proportion of predictions that are correct.

**Formula:**

`Accuracy = Correct Predictions / Total Predictions`

A higher accuracy indicates that the model correctly classified a larger proportion of the evaluated articles.

### 2. Precision

**Precision** measures how many predictions for a particular category were correct.

**Formula:**

`Precision = TP / (TP + FP)`

Where:
- **TP:** True Positives
- **FP:** False Positives

### 3. Recall

**Recall** measures how many of the actual examples belonging to a category were correctly identified.

**Formula:**

`Recall = TP / (TP + FN)`

Where **FN** represents False Negatives.

### 4. F1-Score

The **F1-score** combines precision and recall using their harmonic mean.

**Formula:**

`F1-Score = 2 × (Precision × Recall) / (Precision + Recall)`

A higher F1-score indicates a better balance between precision and recall.

### 5. Macro F1-Score

**Macro F1** calculates the F1-score for each category separately and then takes the average, giving all four categories equal importance.

This metric helps evaluate whether a model performs consistently across the categories.

### 6. Test Loss

**Cross-entropy loss** measures how well the model's predicted probabilities align with the correct labels. Lower loss is generally preferable when comparing compatible evaluation setups.

### 7. Training Time and Parameter Count

- **Training time** estimates the time required for the recorded training process.
- **Parameter count** represents the number of learned numerical values in a model.

Both depend on model architecture, hardware, and training configuration. A larger model does not automatically guarantee better performance.

---

## 🏆 Reference Results

The following metrics are from a previously recorded run of this project. They are **reference results**, not guaranteed outputs from a fresh training run.

| Model | Accuracy | Macro F1-Score |
|---|---:|---:|
| Transformer Encoder | 90.45% | 0.9042 |
| BiLSTM | 91.59% | 0.9156 |
| DistilBERT | 94.59% | 0.9460 |

### Interpretation

- **Transformer Encoder:** demonstrates that an attention-based model trained from scratch can learn useful patterns for news classification.
- **BiLSTM:** demonstrates the effectiveness of bidirectional sequence learning for text classification.
- **DistilBERT:** achieved the strongest accuracy and Macro F1-score in this recorded comparison, potentially benefiting from pretrained language representations.

Fresh training runs may produce different metrics because of changes in random initialization, model configuration, software, hardware, or training settings. Always report the actual metrics measured in the run being discussed.

---

## 📈 Visualizations and Analysis

### 1. Accuracy and Macro F1 Comparison

The comparison chart displays the accuracy and Macro F1-score of all three models. It provides a quick overview of their predictive performance.

### 2. Confusion Matrices

A **confusion matrix** compares the actual categories with the categories predicted by the model.

- **Rows:** actual labels.
- **Columns:** predicted labels.
- **Diagonal cells:** correct predictions.
- **Off-diagonal cells:** incorrect predictions.

For example, a value in the World row and Business column represents a World article incorrectly classified as Business.

Confusion matrices help identify which categories the models frequently confuse.

### 3. Classification Reports

Classification reports show **precision, recall, F1-score, and support** for each category.

**Support** is the number of actual test examples belonging to a category. These reports help identify categories where a model performs particularly well or poorly.

### 4. Training and Validation Loss Curves

Loss curves show how the model's loss changes across training epochs.

- Decreasing training loss indicates that the model is fitting the training data more effectively.
- Decreasing validation loss can indicate improving generalization.
- Decreasing training loss alongside increasing validation loss may indicate overfitting.

Validation curves should only be shown when validation loss has actually been measured.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python** | Main programming language |
| **PyTorch** | Building and training neural networks |
| **Hugging Face Datasets** | Loading the AG News dataset |
| **Hugging Face Transformers** | Loading and fine-tuning DistilBERT |
| **scikit-learn** | Evaluation metrics and confusion matrices |
| **Pandas** | Organizing tabular results |
| **NumPy** | Numerical operations |
| **Matplotlib** | Plotting results |
| **Seaborn** | Statistical visualizations and heatmaps |
| **Google Colab** | Running the notebook, preferably with a GPU |

---

## ▶️ How to Run the Project in Google Colab

### Requirements

- A Google account.
- Internet access to download the dataset and pretrained model.
- A GPU runtime is recommended, particularly for DistilBERT.
- Sufficient RAM and GPU memory for the selected model configuration.

### Steps

1. Open **Google Colab**.
2. Upload the project's `.ipynb` notebook.
3. Select **Runtime → Change runtime type** and choose an available GPU, if possible.
4. Run the dependency-installation cell.
5. Execute the notebook cells in order, starting from the imports and dataset-loading section.
6. Wait for all three models to finish training or fine-tuning.
7. Review the comparison tables, classification reports, and plots.
8. Run the final saving cell to save model weights and results.

### Important Notes

- The first run may take longer because the dataset, packages, and pretrained DistilBERT files may need to download.
- DistilBERT uses pretrained weights; it is not trained entirely from random initialization.
- Colab sessions can disconnect or reset. Download important results or save them to Google Drive.
- Execute cells in order to avoid missing-variable errors.
- Record changes to hyperparameters or training settings when comparing results.

---

## ⚠️ Limitations and Reproducibility

- Model performance can vary due to random initialization, data ordering, hardware, software versions, and optimization settings.
- Setting a random seed improves reproducibility but does not guarantee identical results on every device.
- DistilBERT benefits from pretraining, so the comparison is not based on architecture alone.
- Training time varies depending on hardware and runtime conditions.
- AG News may not represent every publisher, language, or real-world news distribution.
- A confident prediction can still be incorrect.
- Historical results should be labeled clearly and must not be presented as newly measured results.

For reproducibility, document the dataset version, random seed, architecture, batch size, sequence length, learning rate, epochs, tokenizer, pretrained checkpoint, hardware, software versions, and evaluation split.

---

## 📁 Suggested Project Structure

```text
ag-news-classification/
│
├── README.md
├── AG_News_Three_Models_From_Scratch.ipynb
│
├── outputs/
│   ├── model_comparison.csv
│   ├── plots/
│   └── confusion_matrices/
│
└── models/
    ├── transformer_encoder.pt
    ├── bilstm.pt
    └── distilbert_finetuned.pt
```

This is a suggested repository layout. The actual files depend on what the notebook saves. Model checkpoints can be large, so check GitHub's file-size limits before committing them.

---

## 📚 References and Further Learning

1. **Vaswani et al. — Attention Is All You Need**  
   https://arxiv.org/abs/1706.03762

2. **Hochreiter and Schmidhuber — Long Short-Term Memory**  
   https://doi.org/10.1162/neco.1997.9.8.1735

3. **Devlin et al. — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
   https://arxiv.org/abs/1810.04805

4. **Sanh et al. — DistilBERT, a distilled version of BERT**  
   https://arxiv.org/abs/1910.01108

5. **Hugging Face Datasets Documentation**  
   https://huggingface.co/docs/datasets

6. **Hugging Face Transformers Documentation**  
   https://huggingface.co/docs/transformers

7. **PyTorch Documentation**  
   https://pytorch.org/docs/stable/index.html

---

## 🎓 Conclusion

This project compares three deep-learning approaches to news classification: a **Transformer Encoder trained from scratch**, a **BiLSTM trained from scratch**, and **pretrained DistilBERT fine-tuned for the task**.

The project demonstrates how different architectures process language, how pretraining can influence performance, and why evaluating a model requires more than accuracy alone.

By combining accuracy, Macro F1-score, class-wise reports, confusion matrices, and training curves, this project provides a practical introduction to **Natural Language Processing, deep learning, and comparative model evaluation**.
