import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- CLASES CIFAR-10 ---
cifar10_classes = ["avión", "auto", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]

# --- MODELO CIFAR-10 ---
class SimpleCIFAR10Net(torch.nn.Module):
    def __init__(self):
        super(SimpleCIFAR10Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 8 * 8, 128)
        self.fc2 = torch.nn.Linear(128, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCIFAR10Net().to(device)
model.load_state_dict(torch.load("cifar10_simple_model.pth", map_location=device))
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((32, 32))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(img).unsqueeze(0)
    return img

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    imagen_url = None
    if request.method == "POST":
        if "imagen" not in request.files:
            return redirect(request.url)
        file = request.files["imagen"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            img_tensor = preprocess_image(filepath).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(dim=1).item()
                prediccion = cifar10_classes[pred]
            imagen_url = url_for('static', filename=filename)
    return render_template("index.html", prediccion=prediccion, imagen_url=imagen_url)

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)