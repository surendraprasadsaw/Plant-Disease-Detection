import os
from flask import Flask, redirect, render_template, request # type: ignore
from PIL import Image # type: ignore
import torchvision.transforms.functional as TF # type: ignore
import CNN
import numpy as np # type: ignore
import torch # type: ignore
import pandas as pd # type: ignore

# Absolute paths for the CSV files
disease_info_path = r'C:\Users\prasa\Desktop\7TH SEMESTER FINAL PROJECT\Plant-Disease-Detection12\Flask Deployed App\disease_info.csv'
supplement_info_path = r'C:\Users\prasa\Desktop\7TH SEMESTER FINAL PROJECT\Plant-Disease-Detection12\Flask Deployed App\supplement_info.csv'

# Debug paths
print(f"Looking for disease_info.csv at: {os.path.abspath(disease_info_path)}")
print(f"Looking for supplement_info.csv at: {os.path.abspath(supplement_info_path)}")

# Load CSV files
if os.path.exists(disease_info_path) and os.path.exists(supplement_info_path):
    disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
    supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')
    print("Files loaded successfully.")
else:
    raise FileNotFoundError("One or both CSV files not found!")

# Initialize model
model = CNN.CNN(39)
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"), strict=False)
model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

# Flask app setup
app = Flask(__name__)

# Create static/uploads directory
os.makedirs('static/uploads', exist_ok=True)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=image_url, pred=pred, sname=supplement_name, 
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
