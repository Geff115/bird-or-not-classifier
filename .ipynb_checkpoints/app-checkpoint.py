from fastai.vision.all import *
import gradio as gr

# Load model
learn = load_learner('model.pkl')

# Define threshold
CONFIDENCE_THRESHOLD = 0.85

# Inference function
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    confidence = probs[pred_idx].item()
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain – this might not be a bird or forest 🤔", {
            label: float(prob) for label, prob in zip(learn.dls.vocab, probs)
        }

    return pred, {
        label: float(prob) for label, prob in zip(learn.dls.vocab, probs)
    }

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[gr.Label(label="Prediction"), gr.Label(label="Confidence Scores")],
    title="Bird or Not Classifier",
    description="Upload an image to find out if it's a bird 🐦 or not!"
)

interface.launch()