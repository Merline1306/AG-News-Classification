AG News Classification Using Transformer Encoder, BiLSTM, and DistilBERT
A beginner-friendly project comparing three deep-learning approaches for
classifying news articles into World, Sports, Business, and
Sci/Tech.
Table of Contents
1. Project Overview
2. The Problem
3. AG News Dataset
4. Key Concepts
5. The Three Models
6. Comparison
7. Workflow
8. Evaluation Metrics
9. Reference Results
10. Plots and Reports
11. Technologies
12. Run in Google Colab
13. Limitations and Reproducibility
14. Suggested Repository Structure
15. References
    
Project Overview
News websites publish large numbers of articles every day. Automatically
assigning each article to a topic can save time and help organize
information. This project trains and compares three models:
1. Transformer Encoder built from scratch --- randomly initialized
   weights learn from AG News.
2. BiLSTM built from scratch --- a bidirectional recurrent network
   learns from AG News.
3. DistilBERT fine-tuned from pretrained weights --- starts with
   language representations learned during pretraining and adapts them
   to news classification.
The notebook includes training, evaluation, comparison tables,
class-wise reports, confusion matrices, and plots.

The Problem
The task is four-class text classification. Given a news article, the
model predicts one label.
  Category                            What it covers
  World                               International affairs, governments,
                                      conflicts, and world events

  Sports                              Teams, matches, players, and
                                      tournaments

  Business                            Companies, markets, trade, and the
                                      economy

  Sci/Tech                            Science, technology, research, and
                                  computing
A model produces a score for each category. The category with the
highest score is its prediction. Any example scores used in explanations
are illustrative, not measured results.
AG News Dataset
AG News is a standard news-topic classification dataset. The commonly
used Hugging Face version has 120,000 training examples and 7,600 test
examples, across four classes. The notebook loads it programmatically.
- Training set: used to learn model parameters.
- Validation set: used during development to monitor
  generalization and select settings or checkpoints.
- Test set: held aside for final evaluation.
For a trustworthy evaluation, avoid repeatedly tuning the model based on
test-set results.

Key Concepts
Natural Language Processing (NLP)
NLP is the area of AI that enables computers to process human language.
Examples include translation, sentiment analysis, spam detection, and
text classification.
Tokenization
Tokenization splits text into smaller units called tokens. Depending
on the tokenizer, a token may be a word, part of a word, number, or
punctuation mark. For example, "Markets rise today!" could be split
into markets, rise, today, and !.
Vocabulary and token IDs
A vocabulary maps tokens to integers. Neural networks process these IDs
rather than raw words. Unknown tokens need a special unknown-token ID.
Sequences are often padded or truncated to a chosen length so that
examples can be processed in batches.
Embeddings
An embedding maps a token ID to a vector of numbers. During training, a
model learns representations that help it solve the task.
Training, epochs, and loss
During training, the model predicts a label and compares it with the
correct answer using a loss function. An optimizer updates the
model's parameters to reduce the loss. An epoch is one pass through
the training examples. Training loss measures fit to training examples;
validation loss helps monitor performance on examples not used for
parameter updates.
Classification head
The classification head converts a learned article representation into
scores for the four categories.
The Three Models
1. Transformer Encoder (built from scratch)
A Transformer uses attention to let each token gather information
from other tokens. For example, understanding a pronoun may require
looking at a noun earlier in the sentence. Self-attention calculates how
strongly tokens should use information from other tokens.

At a high level, attention forms query, key, and value representations,
calculates attention scores, and combines information according to those
scores. Transformer encoder layers combine attention with feed-forward
processing, residual connections, and normalization.

Components in this project - Embedding layer: maps token IDs to
vectors. - Positional information: helps the model represent token
order. - Encoder layers: use self-attention and feed-forward
networks. - Pooling: combines token representations into one article
representation. - Classification layer: predicts one of the four
labels.

Built from scratch means the model's weights start randomly
initialized rather than from a pretrained language model. PyTorch still
supplies the tensor operations and standard neural-network building
blocks.

Advantages: can model relationships between distant tokens and
process sequence positions in parallel.

Limitations: must learn useful language representations from the
task data; attention can use substantial memory for long sequences;
results depend on architecture and training settings.
2. BiLSTM (built from scratch)
LSTM stands for Long Short-Term Memory, a type of recurrent neural
network designed to carry useful information through a sequence using
gates.

A standard LSTM reads text in one direction. A Bidirectional LSTM
(BiLSTM) uses two LSTMs: - a forward LSTM reads from the beginning to
the end; - a backward LSTM reads from the end to the beginning.

Combining their outputs gives the representation access to context from
both directions.

Components in this project - Embedding layer: maps token IDs to
vectors. - Forward and backward LSTMs: process the sequence in
opposite directions. - Pooling: summarizes the sequence. -
Classification layer: predicts the category.

Advantages: explicitly models sequential information and uses
context from both directions.

Limitations: sequential processing can be slower than parallelizable
approaches; very long sequences can be difficult; the model must learn
from AG News rather than benefiting from pretrained language weights.
3. DistilBERT (pretrained model, fine-tuned)
BERT is a Transformer-based language model pretrained on large text
collections. DistilBERT is a smaller distilled version designed to
retain much of BERT's language capability with fewer layers and lower
computational cost than BERT-base.

Fine-tuning means taking a pretrained model and training it further
for a specific task. Here, DistilBERT's tokenizer prepares the article,
the pretrained model processes it, and a classification head predicts
the four AG News categories. Training adapts the model to the labeled
news examples.

Advantages: benefits from language patterns learned during
pretraining and often performs strongly on classification tasks.

Limitations: requires downloading pretrained weights and tokenizer
files; fine-tuning needs compute; results depend on the checkpoint and
settings.
Comparison
  Feature           Transformer       BiLSTM            DistilBERT
                    Encoder                             
  Architecture      Self-attention    Bidirectional     Distilled
                    encoder           recurrent network pretrained
                                                        Transformer

  Initial weights   Random            Random            Pretrained

  Training on AG    Train from        Train from        Fine-tune
  News              scratch           scratch           pretrained model

  Context           Self-attention    Forward and       Self-attention
                                      backward
                                      recurrence        

  Tokenizer         Project           Project           DistilBERT
                vocabulary        vocabulary        tokenizer
This is not a perfectly controlled comparison of architecture alone:
DistilBERT also benefits from external pretraining, and the models may
differ in size and training cost.
Workflow
1. Load AG News and inspect its labels.
2. Tokenize text and convert it to numerical sequences.
3. Build the custom Transformer Encoder and BiLSTM with random
   initialization.
4. Load DistilBERT and its pretrained tokenizer.
5. Train or fine-tune the models on labeled examples.
6. Monitor loss and validation metrics where implemented.
7. Predict labels for held-out test examples.
8. Calculate accuracy, Macro F1, loss, and class-wise metrics.
9. Generate comparison plots and confusion matrices.
10. Save weights and results if the notebook's save cell is run.
Evaluation Metrics
Accuracy
The fraction of predictions that are correct:

Accuracy = Correct predictions / Total predictions

Accuracy gives an overall view but does not show which classes cause
mistakes.
Precision
Of the examples predicted as a particular class, the fraction that truly
belongs to that class:

Precision = TP / (TP + FP)
Recall
Of all true examples of a class, the fraction the model correctly
identifies:

Recall = TP / (TP + FN)

Here, TP is true positives, FP is false positives, and FN is false
negatives for the class being measured.
F1-score
The harmonic mean of precision and recall:

F1 = 2 × Precision × Recall / (Precision + Recall)
Macro F1
Calculate F1 separately for each of the four classes, then average the
four values equally. This makes each category equally important in the
final score.
Test loss
Cross-entropy loss measures how well predicted class probabilities match
the correct labels. Lower is generally better when evaluation setups are
comparable; interpret it together with accuracy and F1.
Training time and parameter count
Training time depends on hardware, batch size, software, and runtime
load. Parameter count measures learned numerical values; a larger model
is not automatically better.
Reference Results
The following are reference values from a previously recorded run,
not guaranteed outputs of a fresh run.

  Model                   Accuracy   Macro F1
  Transformer Encoder       90.45%     0.9042
  BiLSTM                    91.59%     0.9156
  DistilBERT                94.59%     0.9460

In this recorded comparison, DistilBERT has the highest accuracy and
Macro F1. Its pretrained language representations may help explain its
advantage. The custom models demonstrate how attention-based and
recurrent models perform when trained on the task data from random
initialization.

Always label historical results as historical. For a new run, report the
metrics actually measured by that run rather than replacing them with
these reference values.
Plots and Reports
Accuracy and Macro F1 comparison
A grouped bar chart compares overall test accuracy and Macro F1 for the
three models.
Confusion matrices
Rows represent true labels and columns represent predicted labels.
Diagonal cells are correct predictions; off-diagonal cells show errors.
For example, a value in the World row and Business column represents a
World article predicted as Business.
Classification reports
Reports show precision, recall, F1-score, and support for each category.
Support is the number of true examples in that category in the
evaluated dataset.
Loss curves
Training and validation loss curves help show learning progress. If
training loss decreases but validation loss starts increasing, the model
may be overfitting. Only display validation curves when validation loss
was actually measured.
Technologies
- Python --- programming language.
- PyTorch --- builds and trains custom neural networks.
- Hugging Face Datasets --- loads AG News.
- Hugging Face Transformers --- loads DistilBERT and its
  tokenizer.
- scikit-learn --- computes evaluation metrics and confusion
  matrices.
- Pandas / NumPy --- data and numerical processing.
- Matplotlib / Seaborn --- plots and heatmaps.
- Google Colab --- notebook environment; GPU is recommended.
Run in Google Colab
1. Open Google Colab and upload the .ipynb notebook.
2. Select Runtime → Change runtime type and choose an available GPU
   if possible.
3. Run the dependency-installation cell.
4. Run the notebook cells in order.
5. Wait for each model to finish training or fine-tuning.
6. Inspect the result table, class-wise reports, and plots.
7. Run the save cell to save model weights and the comparison table in
   the current runtime.
The first run may take time because packages, the dataset, and
DistilBERT files may need to download. Colab runtimes can reset, so
download important outputs or save them to Google Drive. Running cells
out of order may cause missing-variable errors.

Limitations and Reproducibility
- Results can change due to random initialization, data ordering, GPU
  operations, software versions, and optimization settings.
- A random seed improves reproducibility but does not guarantee
  identical results on every device.
- DistilBERT benefits from pretraining, while the custom models do
  not; account for this when interpreting the comparison.
- Training time varies by hardware and runtime conditions.
- Benchmark news may not represent all publishers, languages, or
  future news topics.
- High confidence does not guarantee a correct prediction.
- Do not report a metric unless it was measured in the relevant run or
  explicitly labeled as a historical result.
For reproducibility, record the dataset version, seed, model
configuration, batch size, sequence length, epochs, learning rate,
tokenizer/checkpoint, hardware, software versions, and evaluation split.

Suggested Repository Structure
ag-news-classification/
├── README.md
├── AG_News_Three_Models_From_Scratch.ipynb
├── outputs/
│   ├── model_comparison.csv
│   ├── confusion_matrices/
│   └── plots/
└── models/
    ├── transformer_encoder.pt
    ├── bilstm.pt
    └── distilbert_finetuned.pt
This is a suggested layout; exact files depend on what the notebook
saves. Model checkpoints can be large, so check repository storage
limits before committing them. Git LFS or a model-hosting service may be
more appropriate.

Conclusion
This project compares a Transformer Encoder trained from random
initialization, a BiLSTM trained from random initialization, and
pretrained DistilBERT fine-tuned for news classification. The central
lesson is that architecture, pretraining, data preparation, and training
choices all influence performance. Accuracy and Macro F1 summarize
performance, while class-wise reports and confusion matrices reveal
where models succeed and struggle.
