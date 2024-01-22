from fastai.vision.all import *
from fastai.vision.widgets import *
import gradio as gr

# Load learner and get class labels
learn = load_learner('model.pkl')
categories = learn.dls.vocab

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

image = gr.Image()
label = gr.Label()
examples = ['cat.jpg', 'fish.jpg', 'capybara.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False, share=True)