# Character-Level Text Generation using LSTM

**Author:** Anirudh Krishna  
**Institution:** University of Maryland, College Park — MS in Applied Machine Learning  
**Date:** October 23, 2025  

---

## Project Overview

This project implements a **character-level text generation model** using a single-layer **Long Short-Term Memory (LSTM)** network.  
The model is trained on alphabetic sequences to learn **next-character prediction** and generate coherent sequences based on learned patterns.

The objective is to:
- Validate the LSTM’s ability to learn sequential dependencies.
- Demonstrate training stability and convergence on minimal data.
- Generate sample text continuations using the trained model.

---

## Methodology

### Dataset
- The dataset consists of **simple lowercase alphabetic sequences (a–z)**.
- Each input window is mapped to the next expected character.
- The data is tokenized into integer indices and split into train/validation sets.

### Model Architecture
- **Embedding layer** → converts characters into dense representations  
- **LSTM layer** → captures temporal dependencies in sequences  
- **Dense Softmax layer** → outputs probabilities for each next character  

### Training
- **Loss function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Stopping criterion:** Early stopping once validation accuracy ≥ 0.96  
- **Epochs:** 10–15 (typical)  
- **Batch size:** 1–8  

### Hardware Used
- Trained on **NVIDIA Tesla T4 GPU** (Kaggle environment).  
- CuDNN acceleration automatically detected.

---

## Results Summary

| Metric | Value |
|:-------|:------|
| Final Accuracy | **0.9600** |
| Best Epoch | 7 |
| Loss Function | Cross-Entropy |
| Hardware | Tesla T4 GPU |

**Key Observations:**
- LSTM quickly converged and achieved deterministic next-character prediction.
- Greedy decoding generated correct alphabetical continuations.
- Further diversity could be achieved using temperature-based sampling.

---

## Repository Structure

```
 Character-Level-LSTM
 ┣  alphabet-lstm-next-char.py
 ┣  Character_Level_LSTM_Report.pdf
 ┣  README.md
 ┗  assets
```

---

## How to Run

### Option 1: Run on Kaggle or Google Colab
1. Upload the notebook **`alphabet-lstm-next-char.ipynb`**.
2. Ensure GPU acceleration is enabled:
   - Kaggle: *Settings → Accelerator → GPU*
   - Colab: *Runtime → Change runtime type → GPU*
3. Run all cells sequentially (`Runtime → Run all`).
4. The model will automatically:
   - Prepare the alphabet dataset.
   - Train the LSTM.
   - Print training logs and accuracy.
   - Generate sample character predictions.

---

### Option 2: Run Locally (Python Environment)
#### **Step 1 — Install Dependencies**
```bash
pip install tensorflow numpy matplotlib
```

#### **Step 2 — Run the Notebook**
```bash
jupyter notebook alphabet-lstm-next-char.ipynb
```
Then execute each cell to train and test the model.

---

## Output Example

Training log:
```
X_lstm shape: (25, 1, 26) | y_onehot shape: (25, 26)
Reached target accuracy 0.960 at epoch 7. Stopping.
```

Generated sample:
```
Input: 'a' → Predicted Next: 'b'
Input: 'b' → Predicted Next: 'c'
...
```

---

## Reference Report

See the full report: **Character_Level_LSTM_Report.pdf**  
It includes:
- Motivation  
- Model architecture & training setup  
- Experimental results & discussion  
- Academic integrity and AI usage disclosure  

---

## Academic Integrity

I affirm that all modeling, code, and experiments are my own work.  
AI assistance was used solely for report formatting and structure refinement.  
All results and model configurations reflect my own implementation.

---

## Future Improvements

- Introduce **temperature sampling** for diverse text generation.  
- Add **dropout layers** to improve generalization.  
- Explore **multi-layer LSTMs** or **GRU variants** for longer dependencies.

---

**© 2025 Anirudh Krishna — University of Maryland, College Park**
