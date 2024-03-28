from flask import Flask, render_template, request, redirect, url_for, session
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Mock database for user authentication
users = {'username': 'password'}


# ... (Your image processing code remains the same)z

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error='Invalid credentials')

    return render_template('login.html')


# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


# Home route (requires login)
@app.route('/home')
def home():
    if 'username' in session:
        return render_template('home.html', username=session['username'])
    else:
        return redirect(url_for('login'))


# Replace "path_to_root_folder" with your actual root folder containing subfolders
root_folder_path = r"C:\Users\Revanth Ayitha\Desktop\REVANTH FOLDER\galaxy m12\DCIM\Camera"
query_image_path = r"C:\Users\Revanth Ayitha\Desktop\REVANTH FOLDER\galaxy m12\DCIM\Camera\20211013_201204.jpg"


# Function to calculate similarity for folders (remains the same)
def calculate_similarity_for_folders(root_folder, query_image_path):


# Your existing code for similarity calculation

if __name__ == '__main__':
    app.run(debug=True)
