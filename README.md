# PlantCare AI — Plant Leaf Disease Detection

PlantCare AI is a production-grade, user-facing Flask web application designed for agricultural growers, greenhouse operators, and home gardeners to diagnose plant foliar diseases in real time. Powered by a Deep Convolutional Neural Network (CNN) built with TensorFlow/Keras and OpenCV/Pillow, the platform instantly identifies fungal, bacterial, and viral infections from leaf photographs and delivers tailored organic treatments, chemical controls, and fertilizer recommendations.

---

## 🌟 Key Features

- **Leaf Pathology Vision Engine**: Upload leaf photos or snap live pictures via the integrated device webcam.
- **Deep CNN Image Classification**: High-accuracy Convolutional Neural Network classifying multiple major crop species.
- **Actionable Agronomic Prescriptions**:
  - Detailed symptom breakdown
  - Biological pathogen and environmental cause
  - Field hygiene and cultural prevention practices
  - Organic & biological controls (Neem oil, *Bacillus subtilis*, *Trichoderma*)
  - Chemical fungicides and active ingredients (*Mancozeb*, *Copper Oxychloride*, *Metalaxyl*)
- **Agricultural Supplements Directory**: Curated reference guide covering bio-fungicides, chemical protectants, and foliar micronutrients with dosage and harvest intervals.
- **User Diagnosis History**: Secure scan history tracking with timestamps, image thumbnails, confidence ratings, and detailed diagnostic reports.
- **User Authentication**: Secure user registration, session management, and password hashing powered by Werkzeug.
- **Consultation Channel**: Contact agronomists and plant pathology specialists directly through an integrated inquiry system.
- **Clean & Responsive UI**: Built with modern CSS3 and Bootstrap 5, featuring drag-and-drop file upload, instant previews, and mobile-friendly design.
- **Strictly User-Facing**: No administrative overhead, admin routes, or unnecessary bloat.

---

## 📁 Project Structure

```text
plant-disease-detection/
│
├── app.py                      # Core Flask application, routing, and database models
├── config.py                   # App configuration settings and paths
├── requirements.txt            # Python package dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules
│
├── model/
│   ├── plant_disease_model.h5  # Trained TensorFlow/Keras CNN classification model
│   └── class_names.json        # Class mapping and detailed pathology knowledge base
│
├── dataset/                    # Sample leaf image dataset and test specimens
│
├── database/                   # SQLite database (auto-generated plantcare.db)
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom responsive agronomy styles
│   ├── js/
│   │   └── script.js           # Client upload, webcam capture, and UI scripts
│   ├── images/                 # App assets and illustration graphics
│   └── uploads/                # User uploaded leaf images
│
└── templates/
    ├── base.html               # Master layout with navbar and footer
    ├── index.html              # Landing page with workflow and crop overviews
    ├── login.html              # User sign in page
    ├── register.html           # User registration page
    ├── ai_engine.html          # Drag-and-drop & webcam leaf scanner
    ├── result.html             # Detailed diagnostic report & remedies
    ├── history.html            # Scan records & previous diagnosis reports
    ├── supplements.html        # Curated agricultural supplements catalog
    └── contact.html            # Specialist consultation and feedback form
```

---

## 🌾 Supported Crops & Diseases

The CNN classification model supports 20 distinct foliar categories across major staple crops:

1. **Tomato**:
   - Bacterial Spot (*Xanthomonas*)
   - Early Blight (*Alternaria solani*)
   - Late Blight (*Phytophthora infestans*)
   - Leaf Mold (*Passalora fulva*)
   - Septoria Leaf Spot (*Septoria lycopersici*)
   - Tomato Yellow Leaf Curl Virus (TYLCV)
   - Healthy Leaf
2. **Potato**:
   - Early Blight (*Alternaria solani*)
   - Late Blight (*Phytophthora infestans*)
   - Healthy Leaf
3. **Apple**:
   - Apple Scab (*Venturia inaequalis*)
   - Black Rot / Frogeye Leaf Spot (*Botryosphaeria obtusa*)
   - Cedar Apple Rust (*Gymnosporangium*)
   - Healthy Leaf
4. **Corn (Maize)**:
   - Common Rust (*Puccinia sorghi*)
   - Northern Leaf Blight (*Exserohilum turcicum*)
   - Healthy Leaf
5. **Grape**:
   - Black Rot (*Phyllosticta ampelicida*)
   - Leaf Blight / Isariopsis Leaf Spot (*Pseudocercospora*)
   - Healthy Leaf

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- Pip package installer

### 2. Installation
Open your terminal in the `plant-disease-detection` root directory and install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Flask development server:

```bash
python app.py
```

### 4. Access the Web App
Open your web browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## 📷 How to Use the Diagnostic Engine

1. Navigate to **AI Engine** via the navigation bar or the "Scan Leaf Now" button on the home page.
2. Choose one of two capture methods:
   - **Upload an image**: Click "Browse Files" or drag & drop a `.jpg`, `.png`, or `.webp` leaf photo.
   - **Live Camera**: Click "Take Photo", allow browser camera permissions, align the leaf, and click "Capture Photo".
3. Click **Analyze Leaf Disease**.
4. Review the full pathology report:
   - Disease identification & Confidence rating
   - Severity level (Critical, High, Moderate, or Healthy)
   - Biological cause and transmission modes
   - Field prevention and cultural management
   - Organic remedies and approved chemical fungicides

---

## 🔒 Security & Privacy

- All user passwords are encrypted using Werkzeug's secure PBKDF2/SHA256 password hashing.
- File uploads are validated by extension and sanitized using `secure_filename`.
- File sizes are enforced with a 16MB maximum limit (`MAX_CONTENT_LENGTH`).
- SQLite database is cleanly isolated under the `database/` directory.
- No administrative entry points exist, strictly isolating all capabilities to user workflows.

---

## 📄 License
This application is created for agricultural education, farm management, and plant health diagnostics.
