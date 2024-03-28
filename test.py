from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained ResNet50 model
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()


# Define a function to preprocess images
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


# Compute image embeddings
def get_image_embedding(image_path):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        image_embedding = resnet_model(input_image)

    return image_embedding.squeeze().numpy()


# Calculate similarity between two embeddings
def calculate_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))


# Calculate average similarity of an image with all images in a folder
def calculate_avg_similarity_with_folder(query_image_embedding, folder_path):
    total_similarity = 0.0
    num_images = 0

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
                current_embedding = get_image_embedding(image_path)
                similarity = calculate_similarity(query_image_embedding, current_embedding)
                total_similarity += similarity
                num_images += 1

    if num_images > 0:
        avg_similarity = total_similarity / num_images
        return avg_similarity
    else:
        return None


# Function to compare an image against all folders and calculate average similarity
def calculate_similarity_for_folders(root_folder, query_image_path):
    if not os.path.exists(root_folder) or not os.path.isdir(root_folder):
        return None

    query_image_embedding = get_image_embedding(query_image_path)
    folder_similarities = {}

    for root, dirs, files in os.walk(root_folder):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            avg_similarity = calculate_avg_similarity_with_folder(query_image_embedding, folder_path)
            if avg_similarity is not None:
                folder_similarities[dir] = avg_similarity

    return folder_similarities


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        root_folder_path = "./root"#request.form.get('root_folder')
        query_image_path = "P00013_face2.jpg"#request.form.get('image_upload')
        print(root_folder_path)
        print(query_image_path)
        if root_folder_path and query_image_path:
            folder_similarities = calculate_similarity_for_folders(root_folder_path, query_image_path)
            print(folder_similarities)
            return render_template('result.html', folder_similarities=folder_similarities)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
