from fastai.vision.all import *
import gradio as gr

# Load model
learn = load_learner('model.pkl')

# Inference function
def classify_image(img):
    pred, _, probs = learn.predict(img)
    return {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

# Set up the Gradio interface
demo = gr.Interface(
    fn=classify_image, 
    inputs=gr.Image(type='pil'), 
    outputs=gr.Label(), 
    title="Bird or Not Classifier",
    description="Upload an image to find out if it's a bird 🐦 or not!"
)

# Launch the app
demo.launch()