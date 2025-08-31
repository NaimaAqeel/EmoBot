import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load models
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
EMO_THRESHOLD = 0.1  # Increased threshold for more confident detections

# Enhanced emotion mapping with common expressions
EMOTION_MAPPING = {
    "joy": {
        "underlying": ["Contentment", "Pleasure", "Delight", "Happiness"],
        "recommendation": "Savor this positive feeling!",
        "polarity": 1,
        "keywords": ["happy", "joy", "glad", "delighted"]
    },
    "sadness": {
        "underlying": ["Grief", "Loneliness", "Sorrow", "Melancholy"],
        "recommendation": "It's okay to feel this way. Consider talking to someone.",
        "polarity": -1,
        "keywords": ["sad", "unhappy", "depressed", "miserable"]
    },
    "anxiety": {
        "underlying": ["Worry", "Unease", "Apprehension", "Nervousness"],
        "recommendation": "Try box breathing (4-4-4-4) and focus on preparation.",
        "polarity": -1,
        "keywords": ["anxious", "nervous", "worried", "stressed"]
    },
    "fear": {
        "underlying": ["Dread", "Panic", "Terror", "Fright"],
        "recommendation": "Practice grounding techniques to stay present.",
        "polarity": -1,
        "keywords": ["scared", "afraid", "fearful"]
    },
    "anger": {
        "underlying": ["Frustration", "Irritation", "Rage", "Resentment"],
        "recommendation": "Take deep breaths before responding.",
        "polarity": -1,
        "keywords": ["angry", "mad", "furious"]
    },
    "focus": {
        "underlying": ["Concentration", "Attention", "Engagement"],
        "recommendation": "Maintain this productive state with regular breaks.",
        "polarity": 1,
        "keywords": ["focused", "concentrating", "studying"]
    }
}

def create_visualization(emotions):
    fig, ax = plt.subplots(figsize=(8,5))
    colors = []
    for e in emotions.keys():
        if e in EMOTION_MAPPING:
            polarity = EMOTION_MAPPING[e]['polarity']
            color = '#4CAF50' if polarity > 0 else '#F44336'
        else:
            color = '#9E9E9E'  # Neutral for unknown
        colors.append(color)
    
    ax.barh(list(emotions.keys()), list(emotions.values()), color=colors)
    ax.set_xlim(0,1)
    ax.set_xlabel("Confidence Score")
    ax.set_title("Emotion Analysis")
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}">'

def detect_emotions(text):
    # Keyword detection first
    emotions = {}
    text_lower = text.lower()
    
    for emo, data in EMOTION_MAPPING.items():
        for keyword in data['keywords']:
            if keyword in text_lower:
                confidence = 0.9 if any(kw == keyword for kw in text_lower.split()) else 0.7
                if confidence > emotions.get(emo, 0):
                    emotions[emo] = confidence
    
    # Model detection for remaining
    emo_results = emotion_pipeline(text, top_k=None)
    for e in emo_results:
        if e["score"] > EMO_THRESHOLD and e["label"].lower() not in emotions:
            emotions[e["label"].lower()] = e["score"]
    
    return emotions

def analyze_text(text):
    emotions = detect_emotions(text)
    
    details = ""
    positive = []
    negative = []
    
    for i, (emo, score) in enumerate(emotions.items(), 1):
        data = EMOTION_MAPPING.get(emo, {
            "underlying": ["Complex emotion"],
            "recommendation": "This feeling deserves reflection.",
            "polarity": 0
        })
        
        pol = "Positive" if data["polarity"] > 0 else "Negative" if data["polarity"] < 0 else "Neutral"
        
        details += f"Emotion {i}: {emo.capitalize()} ({pol}, Confidence: {score:.2f})\n"
        details += f"Underlying: {', '.join(data['underlying'])}\n"
        details += f"Recommendation: {data['recommendation']}\n\n"
        
        if data["polarity"] > 0: positive.append(emo)
        elif data["polarity"] < 0: negative.append(emo)
    
    conclusion = "\n=== Final Analysis ===\n"
    if positive: conclusion += f"Positive emotions: {', '.join(positive)}\n"
    if negative: conclusion += f"Negative emotions: {', '.join(negative)}\n"
    
    # Enhanced recommendation logic
    if "focus" in positive and not negative:
        final_rec = "Great focus! Maintain this productive state."
    elif positive and negative:
        if "anxiety" in negative:
            final_rec = "Channel your anxiety into constructive planning."
        else:
            final_rec = "Balance these mixed emotions with mindful reflection."
    elif positive:
        final_rec = "Enjoy these positive feelings!"
    elif negative:
        final_rec = "Consider discussing these feelings with someone."
    else:
        final_rec = "The emotional tone is neutral."
    
    conclusion += f"\nFinal Recommendation: {final_rec}"
    
    viz = create_visualization(emotions)
    return details + conclusion, viz

iface = gr.Interface(
    fn=analyze_text,
    inputs=gr.Textbox(label="How are you feeling?", placeholder="I feel..."),
    outputs=[
        gr.Textbox(label="Emotion Breakdown"),
        gr.HTML(label="Emotion Intensity")
    ],
    examples=[
        ["I am happy but at the same time anxious"],
        ["I'm focused on studying but afraid of failure"],
        ["I feel sad and lonely today"],
        ["I'm angry about what happened"]
    ],
    title="EmoBot",
    description="Detects complex emotional states including underlying emotions and provides balanced recommendations."
)

iface.launch()
