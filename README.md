

# EmoBot – Hybrid Emotional State Detection
 Detects complex emotional states including underlying emotions and provides balanced recommendations.

## Overview

**EmoBot** is a **hybrid Gradio-based application** that detects complex emotional states from text input. It combines:

* **Pre-trained NLP models** (DistilBERT emotion classifier)
* **Keyword-based detection** for common expressions

This hybrid approach allows EmoBot to provide:

* Emotion breakdown (e.g., joy, sadness, anxiety, fear, anger, focus)
* Confidence scores for each detected emotion
* Underlying emotional states
* Personalized recommendations based on emotional polarity

The app is interactive and visualizes emotional intensity using horizontal bar charts.

---

## Features

* **Hybrid system**: NLP + keyword detection for robust emotion analysis
* Detects **multiple emotions** in a single text input
* Provides **positive/negative polarity** classification
* Gives **practical recommendations** for managing emotions
* Visualizes **emotion intensity** with a color-coded bar chart
* Includes example inputs for testing mixed emotional states

---

## Demo

Try the **live EmoBot Space** here:
https://huggingface.co/spaces/NaimaAqeel/Sentiment_Analysis

---

## Installation & Usage

### Clone the repository

```bash
git clone https://github.com/NaimaAqeel/EmoBot
cd EmoBot
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run locally

```bash
python app.py
```

* Opens a Gradio interface at `http://127.0.0.1:7860`
* Enter text to get emotional breakdown and recommendations

---

## Requirements

 `requirements.txt` for this project:

```
transformers>=4.12.0
torch>=1.9.0
numpy>=1.21.0
gradio>=3.0
matplotlib>=3.4.0
---

## Files in Repository

* `app.py` – Main Gradio app code (hybrid system)
* `requirements.txt` – Python dependencies
* `README.md` – Project description and usage instructions

---


## Example Inputs

* `"I am happy but at the same time anxious"`
* `"I'm focused on studying but afraid of failure"`
* `"I feel sad and lonely today"`
* `"I'm angry about what happened"`

---

## Notes

* EmoBot uses a **hybrid approach**: DistilBERT pre-trained emotion model + keyword detection
* Keywords enhance detection of common emotional expressions
* Visualizations use **Matplotlib**, encoded to HTML for Gradio display
* Hybrid system improves **accuracy and robustness** for mixed emotions




