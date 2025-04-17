import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import gradio as gr

# Define class names (based on our training folders)
class_names = ['bird', 'forest']

# Define image transforms (must match what fastai used)
transform = T.Compose([
    T.Resize((192, 192)),  # Resize like fastai did (squish to 192x192)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet18 defaults
])

# Define model architecture (must match training)
def get_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    return model

# Load model and weights
model = get_model()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.85

# Prediction function
def predict(image):
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, pred_idx = torch.max(probs, dim=0)
    confidence = confidence.item()
    pred_label = class_names[pred_idx]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain – this might not be a bird or forest 🤔", scores

    return pred_label, scores

# Gradio UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Label(label="Prediction"),
        gr.Label(label="Confidence Scores")
    ],
    title="Bird or Not Classifier",
    description="Upload an image to find out if it's a bird 🐦 or not!"
)

interface.launch()