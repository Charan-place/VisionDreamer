from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from bson import ObjectId

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['flask_login']
users_collection = db['users']
images_collection = db['images']

class User(UserMixin):
    def __init__(self, id, email, password):
        self.id = str(id)
        self.email = email
        self.password = password

class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(id=user_data['_id'], email=user_data['email'], password=user_data['password'])
    except:
        return None

@app.route('/')
def index():
    if current_user.is_authenticated:
        user_id = ObjectId(current_user.id)
        user_images = images_collection.find({"user_id": user_id})
        filenames = [img['filename'] for img in user_images]
        return render_template('home.html', filenames=filenames)
    else:
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        email = form.email.data
        password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user_data = {
            "email": email,
            "password": password
        }
        user_id = users_collection.insert_one(user_data).inserted_id
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data
        user_data = users_collection.find_one({"email": email})
        if user_data and bcrypt.check_password_hash(user_data['password'], password):
            user = User(id=user_data['_id'], email=user_data['email'], password=user_data['password'])
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

def preprocess_image(image, target_size=(1024, 1024)):
    image = image.resize(target_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    return tf.convert_to_tensor(image, dtype=tf.float32)

def calc_loss(img, model):
    img_batch = tf.expand_dims(img, axis=0)
    layer_activations = model(img_batch)
    if len(layer_activations) == 1:
        layer_activations = [layer_activations]
    losses = [tf.reduce_mean(act) for act in layer_activations]
    return tf.reduce_sum(losses)

def deepdream(model, img, step_size=0.01, steps=100):
    img = tf.convert_to_tensor(img)
    for step in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(img)
            loss = calc_loss(img, model)
        gradients = tape.gradient(loss, img)
        gradients /= tf.math.reduce_std(gradients) + 1e-8
        img = img + gradients * step_size
        img = tf.clip_by_value(img, -1, 1)
    return img

def multi_scale_processing(model, image, scales=[1.0, 0.75, 0.5], steps=100, step_size=0.01):
    processed_images = []
    for scale in scales:
        scaled_image = tf.image.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
        dream_image = deepdream(model, scaled_image, step_size=step_size, steps=steps)
        dream_image = tf.image.resize(dream_image, (image.shape[0], image.shape[1]))
        processed_images.append(dream_image)
    return tf.reduce_mean(tf.stack(processed_images), axis=0)

def post_process_image(image_np):
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)

def apply_deepdream(input_image):
    base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    layer_names = ['mixed3', 'mixed5']
    layers = [base_model.get_layer(name).output for name in layer_names]
    dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
    image = preprocess_image(input_image, target_size=(1024, 1024))
    dream_image = multi_scale_processing(dream_model, image, steps=100, step_size=0.01)
    dream_image_np = dream_image.numpy()
    dream_image_pil = post_process_image(dream_image_np)
    resized_image = dream_image_pil.resize((800, 800))  # Adjust dimensions for saving
    
    return resized_image

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Apply DeepDream
        input_image = Image.open(file_path)
        output_image = apply_deepdream(input_image)
        
        # Save the processed image
        output_image_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
        output_image.save(output_image_path)
        
        # Save the processed image info in the database
        images_collection.insert_one({
            "user_id": ObjectId(current_user.id),
            "filename": 'processed_' + file.filename
        })
        
        return redirect(url_for('display_image', filename='processed_' + file.filename))
    
    return 'File upload failed.'

@app.route('/<filename>')
@login_required
def display_image(filename):
    return render_template('result.html', filename=filename)

@app.route('/download/<filename>')
@login_required
def download_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
