'''
BELUM ADA TAMPILANNYA BARU JSON
'''
from flask import Flask, jsonify, request
import io
import torch 
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

app = Flask(__name__)
classes = ('bike','car','helicopter','ship','truck')
model = torch.load("vehicle.pth")


def transform_image(image_bytes):
    transformer = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    image = Image.open(io.BytesIO(image_bytes))
    # x = image.astype(np.float32)
    # x = torch.unsqueeze(x, 0)
    return transformer(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    output = model(tensor)
    x = torch.softmax(output,1)
    x = x.detach().numpy()
    result = np.argmax(x)
    prob = x[0][result]
    pred = classes[result]
    # return pred
    return pred, prob

# with open("split_dataset/cek/000011_02.jpg", "rb") as f:
#     image_bytes = f.read()
#     print (get_prediction(image_bytes=image_bytes))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name, prob = get_prediction(image_bytes=img_bytes)
        # return jsonify({'data': class_name})
        return jsonify({'data': {"result":class_name, 'prob': float(prob)}}) 
    

if __name__ == "__main__":
    app.run()

