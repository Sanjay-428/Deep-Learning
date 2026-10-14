# SMS Spam Detection using GRU (Deep Learning)

A deep learning project that automatically classifies SMS messages as **spam** or **ham** (not spam) using a GRU-based sequential model built with TensorFlow/Keras.

---

## 1. Problem Statement

A mobile communication company wants to identify spam messages automatically using deep learning. A manageable subset of approximately 2,000 labelled messages from the Kaggle SMS Spam Detection dataset is used.

**Dataset:** https://www.kaggle.com/datasets/vishakhdapat/sms-spam-detection-dataset

---

## 2. Tech Stack

| Item | Details |
|---|---|
| Language | Python 3 |
| Platform | Google Colab |
| Deep Learning | TensorFlow / Keras |
| Data handling | pandas, NumPy |
| Evaluation | scikit-learn |
| Visualization | matplotlib, seaborn |

---

## 3. Project Files

| File | Description |
|---|---|
| `spam_sms.csv` | Dataset (columns: `v1` = label, `v2` = message text) |
| `DL test.ipynb` | Google Colab notebook with full code and outputs |
| `README.md` | Project documentation |

---

## 4. Approach (Task-wise)

### Task 1: Load data, handle missing values, preprocess text
- Loaded `spam_sms.csv` using `latin-1` encoding and renamed columns to `label` and `text`.
- Original dataset: **5,572** messages, with **0 missing values**.
- Dropped missing values and removed **duplicate messages** (5,572 to 5,169).
- Took a **stratified subset of 2,000 messages** (Ham: 1,747, Spam: 253), which preserves the original class ratio.
- Text preprocessing:
  - Converted to lowercase
  - Replaced URLs with `url`
  - Replaced numbers with `num`
  - Removed punctuation and special characters
  - Removed extra whitespace
- Encoded labels: `ham = 0`, `spam = 1`.

### Task 2: Tokenization and padded sequences
- Split data into **80% train (1,600)** and **20% test (400)** using stratified sampling.
- Used Keras `Tokenizer` (vocabulary size = 5,000, with an `<OOV>` token for unseen words), fitted on training data only to avoid data leakage.
- Converted messages to integer sequences and padded/truncated them to a fixed length of **50** (`padding='post'`).
- Final shapes: `X_train = (1600, 50)`, `X_test = (400, 50)`.

### Task 3: Model selection and justification (GRU)

**Architecture:**

| Layer | Output Shape | Purpose |
|---|---|---|
| Embedding (5000 x 32) | (None, 50, 32) | Converts word indices into dense vectors |
| GRU (64 units) | (None, 64) | Learns word order and context in the sequence |
| Dropout (0.5) | (None, 64) | Reduces overfitting |
| Dense (32, ReLU) | (None, 32) | Feature learning |
| Dense (1, Sigmoid) | (None, 1) | Spam probability (0 to 1) |

Total parameters: **180,929**. Loss: binary cross-entropy. Optimizer: Adam.

**Why GRU?**
- SMS text is sequential data, so a recurrent model that respects word order is suitable.
- **Simple RNN** suffers from the vanishing gradient problem and forgets earlier words in a sequence.
- **LSTM** solves vanishing gradients but has three gates plus a cell state, so it has more parameters and trains slower.
- **GRU** uses only two gates (update and reset). It handles long-term dependencies like LSTM, but with fewer parameters and faster training. This makes it less prone to overfitting on a small dataset of about 2,000 messages.

### Task 4: Training and evaluation
- Trained with `batch_size = 32`, up to 30 epochs, `validation_split = 0.1`.
- Used **EarlyStopping** (`monitor = val_loss`, `patience = 5`, `restore_best_weights = True`) to stop training at the best epoch and avoid overfitting.
- Evaluated on the 400 unseen test messages.

### Task 5: Confusion matrix and predictions on unseen SMS
- Displayed the confusion matrix as a heatmap.
- Built a `predict_sms()` function that applies the same cleaning, tokenization and padding to a new message, then returns the label and spam probability.

---

## 5. Results

Results may vary slightly between runs (about 1 to 2%) due to random weight initialization.

### Evaluation metrics (test set, 400 messages)

| Metric | Score |
|---|---|
| Accuracy | 0.9825 |
| Precision | 0.9231 |
| Recall | 0.9412 |
| F1-score | 0.9320 |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Ham | 0.99 | 0.99 | 0.99 | 349 |
| Spam | 0.92 | 0.94 | 0.93 | 51 |

### Confusion matrix

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | 345 (TN) | 4 (FP) |
| **Actual Spam** | 3 (FN) | 48 (TP) |

- 393 out of 400 test messages were classified correctly.
- 48 of 51 spam messages were detected.
- Only 4 genuine messages were wrongly flagged as spam.

**Why not accuracy alone?** The dataset is imbalanced (about 87% ham), so a model that always predicts "ham" would already score about 87% accuracy. Precision, recall and F1-score give a fairer picture of spam detection quality.

### Predictions on unseen SMS messages

| Message | Prediction | Spam Probability |
|---|---|---|
| Congratulations! You have won a free iPhone. Click http://claim-now.com to collect your prize | SPAM | 0.993 |
| URGENT! Your account is blocked. Call 09061234567 now to claim 1000 cash | SPAM | 0.993 |
| Hey, are we meeting at the canteen after class? | HAM | 0.000 |
| Please send me the DL notes before tomorrow | HAM | 0.002 |
| WINNER!! You have been selected for a 500 reward. Text CLAIM to 80082 | SPAM | 0.993 |
| Amma said dinner ready, come fast | HAM | 0.000 |

All 6 unseen messages were classified correctly.

---

## 6. How to Run (Google Colab)

1. Open https://colab.research.google.com and create a new notebook.
2. Extract the downloaded Kaggle zip and upload `spam_sms.csv` using the Files panel (upload icon).
3. Paste the cells in order and run them with `Shift + Enter`:

| Cell | Purpose | Task |
|---|---|---|
| Cell 1 | Imports and file path check | Setup |
| Cell 2 | Load data, missing values, 2,000 subset | Task 1 |
| Cell 3 | Text preprocessing | Task 1 |
| Cell 4 | Tokenization and padding | Task 2 |
| Cell 5 | Build GRU model | Task 3 |
| Cell 6 | Train model | Task 4 |
| Cell 7 | Training accuracy/loss graphs | Extra |
| Cell 8 | Metrics and confusion matrix | Task 4 and 5 |
| Cell 9 | Predictions on unseen SMS | Task 5 |

4. If the session disconnects, use **Runtime > Run all** and upload `spam_sms.csv` again.

---

## 7. Challenges and Fix

During the first run, the model got stuck predicting every message as "ham" (the same probability for all inputs). This happened because of class imbalance combined with early stopping that ended training too soon. It was fixed by:
- Removing the manual class weight
- Increasing EarlyStopping patience from 3 to 5
- Setting a fixed random seed (`tf.keras.utils.set_random_seed(42)`) for reproducible results

---

## 8. Possible Improvements

- Compare Simple RNN, LSTM and GRU performance on the same data.
- Use the full dataset (5,169 unique messages) instead of the 2,000 subset.
- Try a Bidirectional GRU or pre-trained embeddings (GloVe).
- Tune hyperparameters such as `MAX_LEN`, embedding size and GRU units.

---

## 9. Author

**Sanjay**
Team: Zenith Coders
