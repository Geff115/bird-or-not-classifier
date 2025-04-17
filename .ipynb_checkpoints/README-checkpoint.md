# 🐦 Bird or Not Classifier

A lightweight image classifier that tells you whether your image contains a bird, a forest, or neither!

Built using FastAI, Gradio, and Python, this app is ideal for a quick image check — powered by a fine-tuned computer vision model.

## 💡 How It Works

- The model was trained on images of **birds** and **forests** scraped using `duckduckgo_search`.
- I trained a `resnet18` model using FastAI and exported only the model weights to a `model.pth` file (instead of `model.pkl`) for safe deployment on Hugging Face Spaces.
- The app uses a confidence threshold to return `"Uncertain"` if it's not sure the image is a bird or forest.

## 🖼 Try it Out

Upload an image and see the prediction:

- **Bird image** → `"bird"`
- **Forest image** → `"forest"`
- **Other** → `"Uncertain – this might not be a bird or forest 🤔"`

## 🧠 Model Info

- **Trained on a small custom dataset of birds and forest images.**
- **Built with FastAI and exported with learn.export().**
- **Includes a confidence threshold — if the model isn't at least 85% confident, it returns "Uncertain".**

## 📦 Installation

Clone the repo:
```bash
git clone https://github.com/Geff115/bird-or-not-classifier.git
cd bird-or-not-classifier
```

Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🖼️ Usage

Launch the app locally:
```bash
python3 app.py
```
Visit http://127.0.0.1:7860 and upload an image to classify.

## 📁 Project Structure

```bash
bird-or-not-classifier/
├── app.py                # Gradio app
├── model.pkl             # Exported FastAI model
├── requirements.txt      # Dependencies
└── README.md             
```

## 🧪 Example Outputs

Image          |          Prediction
- **🐦 Bird (parrot)    |     bird**
- **🌲 Forest Scene      |    forest**
- **🚗 Random Car Photo   |   Uncertain – this might not be a bird or forest 🤔**

## 🚀 Deployment Notes

> I avoid using `model.pkl` due to unsafe pickle serialization in hosted environments like Hugging Face.  
> Instead, I manually reconstruct the model architecture in `app.py` and load `model.pth`.

## 🔮 Future Plans

- **v2: Train with a third "other" class for better generalization.**
- **Add image augmentation + more diverse training data.**
- **Deploy to Hugging Face Spaces.**

## 📄 License

MIT License — use freely for research, projects, or fun 🕊️