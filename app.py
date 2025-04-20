from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('updated_prediction123.h5')

# Define the medicinal uses dictionary
medicinal_uses = {
    "Aloevera": "(Aloe vera) - Used for skin hydration, healing burns, and treating digestive issues.",
    "Amla": "(Phyllanthus emblica) - Rich in Vitamin C, it boosts immunity and aids digestion.",
    "Amruta Balli": "(Tinospora cordifolia) - Known for boosting immunity and treating fever and diabetes.",
    "Arali": "(Nerium oleander) - Though toxic, it has been used in traditional medicine for heart disease and skin conditions (use with caution).",
    "Ashoka": "(Saraca asoca) - Used in Ayurvedic medicine to treat menstrual disorders and improve skin health.",
    "Ashwagandha": "(Withania somnifera) - Reduces stress, anxiety, and improves vitality.",
    "Avacado": "(Persea americana) - Rich in healthy fats, it's good for heart health and skin.",
    "Bamboo": "(Bambusa spp.) - Used in traditional medicine to treat infections and aid in bone health.",
    "Basale": "(Basella alba) - Helps in treating anemia and boosting immunity due to its high iron content.",
    "Betel": "(Piper betle) - Chewed to improve digestion and treat bad breath.",
    "Betel Nut": "(Areca catechu) - Used as a stimulant and to treat digestive issues.",
    "Brahmi": "(Bacopa monnieri) - Enhances memory and reduces anxiety.",
    "Castor": "(Ricinus communis) - Used as a laxative and to treat skin conditions.",
    "Curry Leaf": "(Murraya koenigii) - Improves digestion and manages diabetes.",
    "Doddapatre": "(Plectranthus amboinicus) - Treats coughs, colds, and skin infections.",
    "Ekka": "(Calotropis gigantea) - Used for treating skin diseases and joint pain.",
    "Ganike": "(Clerodendrum serratum) - Traditionally used to treat respiratory conditions.",
    "Guava": "(Psidium guajava) - Rich in Vitamin C, it boosts immunity and aids digestion.",
    "Geranium": "(Pelargonium graveolens) - Used for its anti-inflammatory and astringent properties in skincare.",
    "Henna": "(Lawsonia inermis) - Used for its cooling properties and to treat skin conditions.",
    "Hibiscus": "(Hibiscus rosa-sinensis) - Used to lower blood pressure and treat hair loss.",
    "Honge": "(Pongamia pinnata) - Used for its anti-inflammatory properties and to treat skin disorders.",
    "Insulin": "(Costus igneus) - Known to help regulate blood sugar levels.",
    "Jasmine": "(Jasminum spp.) - Used in aromatherapy to reduce stress and improve skin health.",
    "Lemon": "(Citrus limon) - Boosts immunity and aids digestion due to its high Vitamin C content.",
    "Lemon Grass": "(Cymbopogon citratus) - Used to relieve pain and inflammation and aid digestion.",
    "Mango": "(Mangifera indica) - Rich in antioxidants, it's beneficial for the skin and digestion.",
    "Mint": "(Mentha spp.) - Relieves digestive issues and freshens breath.",
    "Nagadali": "(Rauvolfia serpentina) - Used to treat high blood pressure and mental disorders.",
    "Neem": "(Azadirachta indica) - Known for its antibacterial properties and used to treat skin conditions.",
    "Nithyapushpa": "(Vinca rosea) - Used in treating diabetes and cancer (under medical supervision).",
    "Nooni": "(Morinda citrifolia) - Known for its immune-boosting and pain-relieving properties.",
    "Papaya": "(Carica papaya) - Aids digestion and improves skin health.",
    "Pepper": "(Piper nigrum) - Used to improve digestion and as an anti-inflammatory.",
    "Pomegranate": "(Punica granatum) - Rich in antioxidants, it promotes heart health and reduces inflammation.",
    "Raktachandini": "(Pterocarpus santalinus) - Used to treat skin disorders and improve blood circulation.",
    "Rose": "(Rosa spp.) - Used in skincare for its anti-inflammatory and antioxidant properties.",
    "Sapota": "(Manilkara zapota) - Boosts energy and aids in digestion.",
    "Tulasi": "(Ocimum sanctum) - Known for its immune-boosting and anti-inflammatory properties.",
    "Wood Sorel": "(Oxalis corniculata) - Used to treat fever, inflammation, and digestive issues."
}

# Define allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_medicinal_use(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    # Assuming the plant names are the keys
    plant_names = list(medicinal_uses.keys())
    plant_name = plant_names[predicted_class_index]
    
    return plant_name, medicinal_uses.get(plant_name, "No medicinal use information available.")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        plant_name, uses = get_medicinal_use(file_path)
        
        return jsonify({'prediction': f'{plant_name}: {uses}'})
    
    return jsonify({'error': 'Invalid file format'})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

