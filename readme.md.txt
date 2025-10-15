# 🧠 Small Language Model (SLM) from Scratch

This project implements a **Small Language Model (SLM)** from scratch in Python, following the tutorial by [Shravan Koninti](https://medium.com/@shravankoninti/build-a-small-language-model-slm-from-scratch-3ddd13fa6470).  
It demonstrates how modern language models process, embed, and generate human-like text.

---

## 🚀 Project Overview

The goal of this project is to understand how a language model works at a fundamental level by building one step-by-step — including **tokenization**, **embeddings**, **transformer blocks**, and **training** on small datasets.

---

## 🧩 Key Components

### 1. **Tokenization**
- Uses the `tiktoken` library with GPT-2 encoding.
- Converts raw text into a sequence of numerical tokens.

### 2. **Embedding**
- Maps tokens into dense vectors using an embedding layer.
- Captures semantic meaning of words and phrases.

### 3. **Transformer Block**
- Includes attention mechanisms and feed-forward layers.
- Learns contextual relationships between words.

### 4. **Training**
- Model is trained on a small text dataset.
- Uses loss functions like **cross-entropy** for optimization.

### 5. **Text Generation**
- After training, the model predicts the next token based on previous context.
- Capable of generating short coherent sentences.

---

## ⚙️ How to Run

### **1. Clone this repository**
```bash
git clone https://github.com/your-username/Small-Language-Model.git
cd Small-Language-Model

2️⃣ Install dependencies
pip install -r requirements.txt


If you don’t have a requirements.txt, install manually:

pip install torch numpy tiktoken tqdm

3️⃣ Run the model
python slm.py

🚀 How to Use
🔹 Prepare the data
python slm.py --prepare_data


This tokenizes the text and saves it as train.bin and val.bin.

🔹 Train the model
python slm.py --train


This trains the transformer for a few thousand steps and prints validation loss.

🔹 Generate text
python slm.py --sample


This generates new text sequences based on a given prompt.

🧠 Understanding the Workflow

Tokenization
Converts raw text into numerical tokens using GPT-2’s tokenizer.

Embedding
Maps each token into a dense vector representation.

Self-Attention
Allows the model to learn contextual relationships between tokens.

Training
Uses cross-entropy loss to predict the next token in a sequence.

Generation
Uses sampling to create new text from learned token probabilities.

🧪 Experimentation Ideas

Change block_size, n_embd, or n_layer to observe how complexity affects performance.

Replace tiny_data.txt with your own dataset.

Try different tokenizers or sampling temperatures.

💾 Create Required Files
📘 Create requirements.txt
echo torch > requirements.txt
echo numpy >> requirements.txt
echo tqdm >> requirements.txt
echo tiktoken >> requirements.txt

🧹 Create .gitignore
echo __pycache__/ > .gitignore
echo *.bin >> .gitignore
echo *.pt >> .gitignore
echo tiny_data.txt >> .gitignore

📝 Create README.md
echo # Small Language Model (SLM) from Scratch > README.md
code README.md


Paste this full content and save.

🧭 Git Setup Commands
git init
git add .
git commit -m "Initial commit - Small Language Model project"
git branch -M main
git remote add origin https://github.com/<your-username>/SLM_Project.git
git push -u origin main

🧾 License

This project is open-source and free for educational or research purposes.

👨‍💻 Author

Saksham Bansal, Timothy Pothuraju, Ahmar Aftab
B.Tech CSE | AIML Project