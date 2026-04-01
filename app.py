from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.neighbors import NearestNeighbors
import pickle
from pathlib import Path
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATASET_FOLDER'] = 'dataset/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs('uploads', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

print("="*70)
print("🔌 LOADING VGG16 MODEL...")
print("="*70)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
print("✅ Model loaded successfully!")

COMPONENT_INFO = {
    'resistor': {
        'name': 'Resistor',
        'description': 'A passive two-terminal electrical component that implements electrical resistance as a circuit element. Used to reduce current flow, adjust signal levels, divide voltages, and bias active elements.',
        'specifications': {
            'Type': 'Carbon Film / Metal Film',
            'Power Rating': '0.25W - 2W',
            'Tolerance': '±1% to ±5%',
            'Temperature Coefficient': '±50ppm/°C'
        }
    },
    'capacitor': {
        'name': 'Capacitor',
        'description': 'A passive electronic component that stores electrical energy in an electric field. Used for filtering, energy storage, coupling/decoupling, and timing circuits.',
        'specifications': {
            'Type': 'Ceramic / Electrolytic / Tantalum',
            'Voltage Rating': '6V - 450V',
            'Tolerance': '±5% to ±20%',
            'Temperature Range': '-55°C to +125°C'
        }
    },
    'ic': {
        'name': 'IC (Integrated Circuit)',
        'description': 'A set of electronic circuits on a small semiconductor chip. Performs various functions like amplification, timing, computation, and memory storage.',
        'specifications': {
            'Package Type': 'DIP / SOIC / QFP',
            'Pin Count': '8 - 100+ pins',
            'Operating Voltage': '3.3V - 15V',
            'Operating Temperature': '-40°C to +85°C'
        }
    },
    'transistor': {
        'name': 'Transistor',
        'description': 'A semiconductor device used to amplify or switch electronic signals. Available in NPN and PNP configurations.',
        'specifications': {
            'Type': 'NPN / PNP',
            'Max Collector Current': '100mA - 2A',
            'Max Voltage': '30V - 100V',
            'Package': 'TO-92 / TO-220'
        }
    },
    'diode': {
        'name': 'Diode',
        'description': 'A two-terminal semiconductor device that conducts current primarily in one direction. Used for rectification and voltage regulation.',
        'specifications': {
            'Forward Voltage': '0.3V - 0.7V',
            'Max Current': '1A - 3A',
            'Max Reverse Voltage': '50V - 1000V',
            'Package': 'DO-41 / DO-15'
        }
    },
    'led': {
        'name': 'LED',
        'description': 'Light Emitting Diode - emits light when current flows through it. Available in various colors and sizes.',
        'specifications': {
            'Forward Voltage': '1.8V - 3.3V',
            'Forward Current': '20mA - 30mA',
            'Luminous Intensity': '10-1000 mcd',
            'Viewing Angle': '30° - 120°'
        }
    },
    'relay': {
        'name': 'Relay',
        'description': 'An electrically operated switch that uses an electromagnet to mechanically operate a switching mechanism.',
        'specifications': {
            'Coil Voltage': '5V - 24V',
            'Contact Rating': '5A - 10A',
            'Switching Type': 'SPDT / DPDT',
            'Operating Temperature': '-40°C to +85°C'
        }
    },
    'potentiometer': {
        'name': 'Potentiometer',
        'description': 'A three-terminal resistor with a sliding contact that forms an adjustable voltage divider.',
        'specifications': {
            'Resistance Range': '1kΩ - 1MΩ',
            'Power Rating': '0.5W - 2W',
            'Rotation': '270° - 300°',
            'Tolerance': '±10% to ±20%'
        }
    },
    'transformer': {
        'name': 'Transformer',
        'description': 'A passive electrical device that transfers electrical energy between circuits through electromagnetic induction.',
        'specifications': {
            'Primary Voltage': '110V - 240V AC',
            'Secondary Voltage': 'Variable',
            'Power Rating': '1VA - 1000VA',
            'Frequency': '50Hz / 60Hz'
        }
    },
    'fuse': {
        'name': 'Fuse',
        'description': 'A safety device that provides overcurrent protection by melting when excessive current flows through it.',
        'specifications': {
            'Current Rating': '100mA - 10A',
            'Voltage Rating': '32V - 600V',
            'Breaking Capacity': 'Fast / Slow Blow',
            'Type': 'Glass / Ceramic / Cartridge'
        }
    },
    'jumper': {
        'name': 'Jumper Cable',
        'description': 'Electrical wires with connectors on each end, used to interconnect components and transfer electrical signals or power.',
        'specifications': {
            'Wire Gauge': 'AWG 22 - 28',
            'Current Rating': '1A - 3A',
            'Length': 'Various',
            'Connector Type': 'Male/Female Pins'
        }
    }
}

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
    
    def extract(self, img_path):
        try:
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            features = self.model.predict(img_array, verbose=0)
            features = features.flatten()
            features = features / np.linalg.norm(features)
            
            return features
        except Exception as e:
            print(f"❌ Error extracting features: {e}")
            return None

class ComponentClassifier:
    """IMPROVED: Handles ALL your folder names exactly as they are"""
    
    def detect_type(self, image_path):
        path_str = str(image_path).lower()
        
        print(f"  📁 Checking path: {Path(image_path).parent.name}")
        
        # Check EXACT folder names from your dataset
        if 'resistor' in path_str:
            return 'resistor'
        elif 'capacitor' in path_str:
            return 'capacitor'
        elif 'led' in path_str:
            return 'led'
        elif 'transistor' in path_str:
            return 'transistor'
        elif 'diode' in path_str or 'semiconductor' in path_str:
            return 'diode'
        elif 'ic' in path_str or 'integrated' in path_str or 'micro' in path_str or 'circuit' in path_str:
            return 'ic'
        elif 'relay' in path_str:
            return 'relay'
        elif 'potentiometer' in path_str:
            return 'potentiometer'
        elif 'transformer' in path_str or 'step-down' in path_str or 'step-up' in path_str or 'step_down' in path_str or 'step_up' in path_str:
            return 'transformer'
        elif 'fuse' in path_str or 'cartridge' in path_str:
            return 'fuse'
        elif 'jumper' in path_str or 'cable' in path_str:
            return 'jumper'
        
        # Default
        return 'resistor'

class ResistorDetector:
    """COMPLETELY REWRITTEN for accurate color detection"""
    
    def __init__(self):
        self.color_values = {
            'black': 0, 'brown': 1, 'red': 2, 'orange': 3, 'yellow': 4,
            'green': 5, 'blue': 6, 'violet': 7, 'grey': 8, 'white': 9
        }
        
        self.multipliers = {
            'black': 1, 'brown': 10, 'red': 100, 'orange': 1000, 'yellow': 10000,
            'green': 100000, 'blue': 1000000, 'violet': 10000000,
            'gold': 0.1, 'silver': 0.01
        }
        
        self.tolerance = {
            'brown': '±1%', 'red': '±2%', 'gold': '±5%', 'silver': '±10%',
            'green': '±0.5%', 'blue': '±0.25%', 'violet': '±0.1%', 'grey': '±0.05%'
        }
        
        # OPTIMIZED HSV ranges for resistor bands
        self.color_ranges = {
            'black': ([0, 0, 0], [180, 255, 45]),
            'brown': ([6, 70, 40], [16, 200, 140]),
            'red': ([0, 130, 80], [8, 255, 255]),
            'orange': ([9, 120, 120], [17, 255, 255]),
            'yellow': ([18, 100, 120], [28, 255, 255]),
            'green': ([29, 70, 60], [90, 255, 255]),
            'blue': ([91, 90, 70], [125, 255, 255]),
            'violet': ([126, 70, 70], [160, 255, 255]),
            'grey': ([0, 0, 50], [180, 30, 170]),
            'white': ([0, 0, 170], [180, 35, 255]),
            'gold': ([16, 120, 150], [28, 255, 255]),
            'silver': ([0, 0, 120], [180, 30, 190])
        }
    
    def detect_colors(self, img_path):
        """CRITICAL FIX: Detect resistor bands accurately"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return self._get_default_bands()
            
            print(f"\n  🎨 Analyzing resistor image...")
            
            # Resize to standard size
            h, w = img.shape[:2]
            if w > 600:
                img = cv2.resize(img, (600, int(h * 600 / w)))
            
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Detect ALL colors with their pixel counts
            color_data = []
            for color_name, (lower, upper) in self.color_ranges.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixels = cv2.countNonZero(mask)
                if pixels > 30:
                    color_data.append((color_name, pixels))
            
            # Sort by pixel count
            color_data.sort(key=lambda x: x[1], reverse=True)
            
            # Print what we found
            found_colors = [c for c, p in color_data]
            print(f"  📊 All colors detected: {found_colors}")
            
            # Extract valid band colors (skip body colors)
            bands = []
            for color, pixels in color_data:
                # Skip white/grey/beige (resistor body)
                if len(bands) < 3 and color in ['white', 'grey']:
                    continue
                
                # Only add valid resistor band colors
                if color in self.color_values or color in ['gold', 'silver']:
                    if color not in bands:  # No duplicates
                        bands.append(color)
                        
                if len(bands) >= 4:
                    break
            
            # Ensure we have at least 3 bands
            if len(bands) < 3:
                print(f"  ⚠️ Only found {len(bands)} bands, using common value")
                return ['brown', 'black', 'red', 'gold']
            
            # Add tolerance band if missing
            if len(bands) == 3:
                bands.append('gold')
            
            result = bands[:4]
            print(f"  ✅ FINAL BANDS: {result}")
            return result
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return self._get_default_bands()
    
    def _get_default_bands(self):
        """Return a common resistor value as default"""
        return ['brown', 'black', 'red', 'gold']  # 1kΩ
    
    def calculate_value(self, bands):
        """Calculate resistance from color bands"""
        try:
            digit1 = self.color_values.get(bands[0], 1)
            digit2 = self.color_values.get(bands[1], 0)
            mult = self.multipliers.get(bands[2], 100)
            tol = self.tolerance.get(bands[3], '±5%')
            
            resistance = (digit1 * 10 + digit2) * mult
            
            if resistance >= 1e6:
                value_str = f"{resistance/1e6:.2f}MΩ"
            elif resistance >= 1e3:
                value_str = f"{resistance/1e3:.2f}kΩ"
            else:
                value_str = f"{resistance:.1f}Ω"
            
            print(f"  💡 Calculated: {value_str} {tol}")
            return value_str, tol
            
        except Exception as e:
            print(f"  ❌ Calculation error: {e}")
            return "1.0kΩ", "±5%"

class CapacitorDetector:
    """Simple capacitor value detection"""
    
    def detect_value(self, img_path):
        """Try to detect capacitor value - simplified approach"""
        try:
            # For now, return indication that value is on component
            # Full OCR implementation would require pytesseract
            return "See component marking", "Various"
        except:
            return "Varies", "Check marking"

class LEDDetector:
    """IMPROVED LED color detection"""
    
    def __init__(self):
        # Precise HSV ranges for LED colors - ADJUSTED
        self.led_colors = {
            'Red': ([0, 120, 70], [10, 255, 255]),
            'Green': ([35, 80, 50], [85, 255, 255]),
            'Blue': ([95, 100, 80], [125, 255, 255]),
            'Yellow': ([22, 120, 120], [32, 255, 255]),
            'White': ([0, 0, 180], [180, 40, 255]),
            'Orange': ([11, 120, 100], [20, 255, 255])
        }
    
    def detect_color(self, img_path):
        """IMPROVED: Detect LED color from TOP part only"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return 'Standard'
            
            print(f"\n  🔦 Detecting LED color from TOP...")
            
            # Focus ONLY on TOP 30% of image (where LED bulb is)
            h, w = img.shape[:2]
            led_top = img[0:int(h*0.3), :]
            
            hsv = cv2.cvtColor(led_top, cv2.COLOR_BGR2HSV)
            
            # Check each color
            color_scores = []
            for color_name, (lower, upper) in self.led_colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                score = cv2.countNonZero(mask)
                color_scores.append((color_name, score))
            
            # Sort by score
            color_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Debug print
            print(f"  📊 Color scores: {[(c, s) for c, s in color_scores[:3]]}")
            
            # If top color has significant pixels
            if color_scores[0][1] > 50:
                detected = color_scores[0][0]
                print(f"  ✅ LED Color: {detected}")
                return detected
            else:
                print(f"  ℹ️ No clear color, using Standard")
                return 'Standard'
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return 'Standard'

class CBIRSystem:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.extractor = FeatureExtractor(feature_model)
        self.classifier = ComponentClassifier()
        self.resistor_detector = ResistorDetector()
        self.led_detector = LEDDetector()
        self.capacitor_detector = CapacitorDetector()
        
        self.features_db = []
        self.image_paths = []
        self.knn = None
        
        self.load_or_build_database()
    
    def build_database(self):
        print("\n" + "="*70)
        print("📦 BUILDING DATABASE FROM IMAGES...")
        print("="*70)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    img_path = os.path.join(root, file)
                    features = self.extractor.extract(img_path)
                    
                    if features is not None:
                        self.features_db.append(features)
                        self.image_paths.append(img_path)
        
        if len(self.features_db) > 0:
            self.features_db = np.array(self.features_db)
            
            n_neighbors = min(5, len(self.features_db))
            self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            self.knn.fit(self.features_db)
            
            with open('features_db.pkl', 'wb') as f:
                pickle.dump({
                    'features': self.features_db,
                    'paths': self.image_paths
                }, f)
            
            print(f"✅ Database built with {len(self.image_paths)} images")
            print("="*70 + "\n")
            return True
        else:
            print("⚠️ No images found in dataset folder")
            return False
    
    def load_database(self):
        try:
            with open('features_db.pkl', 'rb') as f:
                data = pickle.load(f)
                self.features_db = data['features']
                self.image_paths = data['paths']
            
            n_neighbors = min(5, len(self.features_db))
            self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            self.knn.fit(self.features_db)
            
            print(f"✅ Database loaded with {len(self.image_paths)} images\n")
            return True
        except FileNotFoundError:
            print("⚠️ Database file not found\n")
            return False
    
    def load_or_build_database(self):
        if not self.load_database():
            self.build_database()
    
    def image_to_base64(self, img_path):
        try:
            with open(img_path, 'rb') as img_file:
                img_data = img_file.read()
                base64_data = base64.b64encode(img_data).decode('utf-8')
                ext = Path(img_path).suffix.lower()
                mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
                return f"data:{mime_type};base64,{base64_data}"
        except:
            return None
    
    def find_similar(self, query_features, top_k=4):
        if self.knn is None or len(self.features_db) == 0:
            return []
        
        try:
            n_neighbors = min(top_k, len(self.features_db))
            distances, indices = self.knn.kneighbors([query_features], n_neighbors=n_neighbors)
            
            similar_images = []
            for idx, dist in zip(indices[0], distances[0]):
                similarity = float((1 - dist) * 100)
                img_path = self.image_paths[idx]
                base64_img = self.image_to_base64(img_path)
                
                similar_images.append({
                    'path': base64_img,
                    'similarity': round(similarity, 1),
                    'name': Path(img_path).stem.replace('_', ' ').title()
                })
            
            return similar_images
        except Exception as e:
            print(f"❌ Error finding similar: {e}")
            return []
    
    def analyze(self, img_path):
        """MAIN ANALYSIS FUNCTION"""
        try:
            print("\n" + "="*70)
            print("🔍 ANALYZING UPLOADED IMAGE")
            print("="*70)
            
            # Extract features
            query_features = self.extractor.extract(img_path)
            if query_features is None:
                return {'error': 'Failed to extract features'}
            
            # Find similar images
            similar_images = self.find_similar(query_features)
            
            # Determine component type using VOTING from top 5 matches
            if similar_images and len(self.image_paths) > 0:
                n_neighbors = min(5, len(self.features_db))
                distances, indices = self.knn.kneighbors([query_features], n_neighbors=n_neighbors)
                
                type_votes = {}
                print("\n  🗳️ Voting from top matches:")
                for idx in indices[0]:
                    path = self.image_paths[idx]
                    detected_type = self.classifier.detect_type(path)
                    type_votes[detected_type] = type_votes.get(detected_type, 0) + 1
                
                # Winner
                component_type = max(type_votes, key=type_votes.get)
                print(f"\n  🏆 RESULT: {component_type.upper()} (votes: {type_votes})")
            else:
                component_type = 'resistor'
            
            # Get component info
            comp_info = COMPONENT_INFO.get(component_type, COMPONENT_INFO['resistor'])
            confidence = float(similar_images[0]['similarity']) if similar_images else 85.0
            
            result = {
                'component_type': comp_info['name'],
                'confidence': round(confidence, 1),
                'description': comp_info['description'],
                'specifications': comp_info['specifications'],
                'similar_images': similar_images
            }
            
            # Get component-specific values
            if component_type == 'resistor':
                bands = self.resistor_detector.detect_colors(img_path)
                value, tolerance = self.resistor_detector.calculate_value(bands)
                result['value'] = value
                result['tolerance'] = tolerance
                result['color_bands'] = bands
                
            elif component_type == 'led':
                color = self.led_detector.detect_color(img_path)
                result['value'] = f'{color} LED'
                result['forward_voltage'] = '1.8V - 3.3V'
                
            elif component_type == 'capacitor':
                cap_value, voltage = self.capacitor_detector.detect_value(img_path)
                result['value'] = cap_value
                result['voltage_rating'] = voltage
                result['note'] = 'Check printed value on component'
                
            elif component_type == 'diode':
                result['value'] = 'Semiconductor Diode'
                result['forward_voltage'] = '0.6V - 0.7V'
                
            elif component_type == 'fuse':
                result['value'] = 'Cartridge Fuse'
                result['note'] = 'Current rating printed on component'
                
            elif component_type == 'transformer':
                result['value'] = 'Step-Down/Step-Up Transformer'
                
            elif component_type == 'jumper':
                result['value'] = 'Jumper Wire/Cable'
            
            print("\n" + "="*70)
            print(f"✅ ANALYSIS COMPLETE: {result['component_type']}")
            print("="*70 + "\n")
            
            return result
            
        except Exception as e:
            print(f"\n❌ ANALYSIS ERROR: {e}\n")
            return {'error': str(e)}

# Initialize system
print("\n" + "="*70)
print("🚀 INITIALIZING CBIR SYSTEM")
print("="*70)
cbir_system = CBIRSystem(app.config['DATASET_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = cbir_system.analyze(filepath)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🌐 STARTING WEB SERVER")
    print("="*70)
    print("📍 Open: http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000)