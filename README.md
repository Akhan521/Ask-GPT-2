# 🤖 Ask-GPT-2: Fine-Tuning GPT-2 for Helpful Question Answering

I was curious about how language models can be steered toward specific tasks and become more helpful with just a bit of guidance. I chose question answering because it's a practical task that highlights how well a model understands and responds to prompts, perfect for testing whether fine-tuning actually makes a model more helpful.

My project explores how to fine-tune a small, general-purpose language model (GPT-2) to perform basic question answering using the [Guanaco QA dataset](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k). My goal was to understand how even compact models can be guided to excel at specific tasks through fine-tuning.

> Fine-tuning allows models to adapt to new tasks (like question answering), improve response relevance, and become more aligned with user intent, all without massive compute or data.
---

## 📌 What This Project Covers

-  Fine-tuned `distilgpt2` on a small Q&A dataset
-  Compared outputs before and after fine-tuning
-  Packaged a Colab demo for quick experimentation
-  Deployed model to Hugging Face Hub for public access

---

## 💡 Why This Matters

GPT-2 was originally trained to complete text, not answer questions directly. Without fine-tuning, it struggles with structure, relevance, and clarity. My project highlights how targeted examples can guide it toward more helpful behavior.

Before fine-tuning, GPT-2 tends to:
- Continue the question as if it's part of a story, rather than answering it
- Give vague or unrelated completions
- Repeat keywords from the prompt without providing meaningful information

---

## 🧠 What I Learned

- How to fine-tune Hugging Face models using `trl` and `SFTTrainer`
- How to evaluate output differences from training
- How to manage model loading and deployment
- How to structure a reproducible ML workflow for others to try

---

## 🚀 Try It Yourself

You can play with the base and fine-tuned models in this interactive Colab demo:

👉 [Colab Demo Notebook](https://colab.research.google.com/drive/1mIY6XrOPOAuhILn_oL9H4j_Y_bPvQio-?usp=sharing)  

---

## 🛠 How It Works

My codebase is modular and quite straightforward:

- `train.py`: fine-tunes the model using Hugging Face's TRL (Transformer Reinforcement Learning) library
- `inference.py`: run a prompt with either the base or fine-tuned model
- `generate.py`: handles generation configuration, text generation, and sampling
- `model.py`, `data_loader.py`: utilities for loading models and datasets

Upon executing `train.py`, your trained model artifacts will be stored in the `logs/` directory locally.
> If you wish to simply try out my fine-tuned model (no training necessary), you can do so via my publicly-accessible model on 🤗 Hub.

---

## ⚠️ Limitations & Next Steps

While fine-tuning helps, this is still a small model with limited reasoning power:

- Answers may lack accuracy or depth
- No formal evaluation metrics are included
- Full fine-tuning was used; LoRA or QLoRA could reduce compute costs
- Dataset is small (~1k examples)

Future improvements to my project could explore smarter evaluation, parameter-efficient tuning, or comparing other model architectures.

---
## 🧑‍💻 Getting Started Locally

To run this project on your own machine, follow these steps below:

### ✅ Prerequisites

- Python
- Git
- A GPU (recommended for training, not required)
- `pip` or `virtualenv` (recommended)

### 🛠️ Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/Akhan521/Ask-GPT-2.git
cd Ask-GPT-2
```

#### 2. (Optional) Create + Activate a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # On Mac: source .venv/bin/activate
```

#### 3. Install Project Dependencies
```bash
# All necessary modules for running this project.
pip install -r requirements.txt
```

#### Option A: Fine-Tune Locally
```bash
# This will fine-tune distilgpt2 on the Guanaco QA dataset
# and save your model + tokenizer inside the logs/ directory.
python -m src/train.py
```

#### Option B: Skip Training - Use Fine-Tuned Model from 🤗 Hub

You can load my already fine-tuned model directly:
 - This is handled automatically in `inference.py`.


#### 4. Run Inference + Compare Outputs
```bash
# Run prompts against both the base and fine-tuned models
# and compare their generated responses side-by-side.
python -m src/inference.py
```
- If you're loading your locally fine-tuned model from `logs/`, be sure to set `use_local_finetuned` to `True` in `inference.py`.
---

## 📬 Reaching Out

Thank you for reading about my project and sharing your time with me!

Feel free to reach out via [GitHub](https://github.com/Akhan521) or connect on [LinkedIn](https://www.linkedin.com/in/aamir-khan-aak521/) if you have any feedback for me or would like to chat with me!

You can also view my portfolio here: [Portfolio](https://aamir-khans-portfolio.vercel.app/)
