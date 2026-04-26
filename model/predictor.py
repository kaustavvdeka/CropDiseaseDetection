"""
CropGuard NE — Inference Engine
Loads the trained .h5 model and classifies a single image.
Can be imported by Streamlit app or used standalone.
"""

import os, json
import numpy as np
from PIL import Image
import tensorflow as tf


# ─── DISEASE METADATA ─────────────────────────────────────────────────────────
DISEASE_META = {
    "Apple___Apple_scab": {
        "cause": "Venturia inaequalis (fungus)",
        "severity_default": "Moderate",
        "symptoms": ["Olive-green to black scab-like spots on leaves", "Velvety lesions on fruit surface", "Premature leaf drop"],
        "northeast_context": "Apple scab is rare in NE India's lowlands but can affect apple orchards in higher elevations of Arunachal Pradesh and Sikkim during wet monsoon seasons.",
        "treatment": ["Spray Mancozeb 75WP @ 2g/L water", "Apply Captan 50WP @ 2.5g/L at 10-day intervals"],
        "organic_options": ["Neem oil spray @ 5ml/L water", "Bordeaux mixture (1%)", "Remove and burn infected leaves"],
        "prevention": ["Avoid overhead irrigation", "Prune for air circulation"],
    },
    "Apple___Black_rot": {
        "cause": "Botryosphaeria obtusa (fungus)",
        "severity_default": "High",
        "symptoms": ["Brown to black circular spots on leaves", "Frog-eye lesions with purple margins", "Mummified fruits"],
        "northeast_context": "Black rot pressure increases in humid conditions typical of Sikkim and Arunachal Pradesh apple belts.",
        "treatment": ["Captan 50WP @ 2.5g/L", "Thiophanate-methyl @ 1g/L"],
        "organic_options": ["Bordeaux mixture 1%", "Remove mummified fruits", "Compost tea spray"],
        "prevention": ["Prune dead wood regularly", "Avoid wounding bark"],
    },
    "Potato___Late_blight": {
        "cause": "Phytophthora infestans (oomycete)",
        "severity_default": "Critical",
        "symptoms": ["Water-soaked dark lesions on leaf edges", "White sporulation on leaf undersides", "Rapid blackening and collapse of foliage", "Dark brown sunken lesions on tubers"],
        "northeast_context": "Potato Late Blight is the #1 threat to potato farmers in Meghalaya (East Khasi Hills), Nagaland, and Manipur. The cool, humid monsoon climate (June–September) creates ideal conditions. Meghalaya alone loses 20–30% of its potato crop annually. Early warning is critical.",
        "treatment": ["Mancozeb 75WP @ 2.5g/L every 7 days", "Metalaxyl+Mancozeb (Ridomil Gold) @ 2.5g/L", "Cymoxanil+Mancozeb @ 3g/L for active infections"],
        "organic_options": ["Copper oxychloride 50WP @ 3g/L", "Trichoderma viride soil application", "Remove and destroy infected haulms immediately"],
        "prevention": ["Use certified disease-free seed tubers", "Avoid waterlogging — ridge planting", "Apply preventive spray before monsoon onset"],
    },
    "Potato___Early_blight": {
        "cause": "Alternaria solani (fungus)",
        "severity_default": "Moderate",
        "symptoms": ["Brown circular spots with concentric rings (target-board pattern)", "Yellow halo around lesions", "Lower leaves affected first"],
        "northeast_context": "Common in all NE states during dry spells between rain events. Affects potato crops in Meghalaya, Mizoram, and Nagaland upland areas.",
        "treatment": ["Mancozeb 75WP @ 2g/L", "Chlorothalonil 75WP @ 2g/L"],
        "organic_options": ["Neem oil @ 5ml/L + soap", "Baking soda spray (1 tsp/L)", "Compost mulch to reduce splash"],
        "prevention": ["Crop rotation (3-year cycle)", "Avoid overhead irrigation"],
    },
    "Potato___healthy": {
        "cause": "None",
        "severity_default": "None",
        "symptoms": ["No visible disease symptoms", "Leaves are dark green and uniform", "Vigorous plant growth"],
        "northeast_context": "Your potato plant appears healthy! Continue monitoring weekly, especially during the humid June–September period in NE India.",
        "treatment": [],
        "organic_options": ["Maintain soil health with compost", "Monitor weekly for early signs"],
        "prevention": ["Preventive neem oil spray monthly", "Proper spacing for air circulation"],
    },
    "Tomato___Bacterial_spot": {
        "cause": "Xanthomonas campestris pv. vesicatoria (bacteria)",
        "severity_default": "Moderate",
        "symptoms": ["Water-soaked spots becoming brown with yellow halo", "Spots coalesce under high humidity", "Raised scabby lesions on fruit"],
        "northeast_context": "Tomato Bacterial Spot thrives in the warm, humid climate of Assam's plains (Kamrup, Sonitpur) and Manipur valley. A significant problem during post-monsoon tomato cultivation.",
        "treatment": ["Copper hydroxide 77WP @ 3g/L", "Streptomycin sulphate @ 0.5g/L (bactericide)"],
        "organic_options": ["Copper-based Bordeaux mixture 1%", "Avoid overhead irrigation", "Remove and compost infected debris"],
        "prevention": ["Use resistant varieties (Arka Vikas, Pusa Ruby)", "Avoid working in wet fields"],
    },
    "Tomato___Early_blight": {
        "cause": "Alternaria solani (fungus)",
        "severity_default": "Moderate",
        "symptoms": ["Dark brown spots with concentric rings on older leaves", "Yellow tissue surrounding lesions", "Stem lesions near soil line (collar rot)"],
        "northeast_context": "Very common in Assam and Tripura tomato gardens. Humid nights and warm days during October–December create optimal conditions.",
        "treatment": ["Mancozeb 75WP @ 2.5g/L", "Iprodione 50WP @ 1g/L"],
        "organic_options": ["Neem oil 2% spray", "Trichoderma harzianum soil drench", "Remove lower infected leaves"],
        "prevention": ["Stake plants to improve airflow", "Mulch to reduce soil splash"],
    },
    "Tomato___Late_blight": {
        "cause": "Phytophthora infestans (oomycete)",
        "severity_default": "Critical",
        "symptoms": ["Large irregular dark lesions on leaves", "White mold on underside in humid conditions", "Rapid plant collapse", "Dark greasy fruit lesions"],
        "northeast_context": "Devastating during the monsoon season across all NE states. Tomato fields in Meghalaya's Ri-Bhoi and Assam's hill districts are most affected.",
        "treatment": ["Metalaxyl+Mancozeb @ 2.5g/L preventively", "Fenamidone+Mancozeb @ 3g/L curatively"],
        "organic_options": ["Copper oxychloride @ 3g/L", "Destroy infected plants immediately", "Potassium bicarbonate spray"],
        "prevention": ["Avoid planting in poorly drained areas", "Use mulch film"],
    },
    "Tomato___healthy": {
        "cause": "None",
        "severity_default": "None",
        "symptoms": ["Leaves are deep green with no lesions", "Stems are sturdy and upright", "No unusual spots or discoloration"],
        "northeast_context": "Your tomato plant is healthy! Assam and Tripura's tomato season (Oct–Feb) is peak time for disease pressure — maintain weekly monitoring.",
        "treatment": [],
        "organic_options": ["Weekly neem oil spray as preventive", "Compost tea soil drench"],
        "prevention": ["Monitor for whitefly (TYLCV vector)", "Maintain 45–60 cm plant spacing"],
    },
}

# Default metadata for classes not explicitly listed
DEFAULT_META = {
    "cause": "See disease-specific literature",
    "severity_default": "Moderate",
    "symptoms": ["Consult local KVK (Krishi Vigyan Kendra) for symptom verification"],
    "northeast_context": "Contact your nearest KVK in Assam, Meghalaya, Manipur, Nagaland, Arunachal Pradesh, Tripura, Mizoram, or Sikkim for localized advice.",
    "treatment": ["Consult ICAR-NRC for specific recommendations"],
    "organic_options": ["Neem-based formulations are broadly effective", "Contact KVK for local organic options"],
    "prevention": ["Crop rotation", "Use certified disease-free seeds or planting material"],
}


class PlantDiseasePredictor:
    """Loads trained model and provides prediction with full metadata."""

    def __init__(self, model_path: str, class_indices_path: str):
        print(f"[Predictor] Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        with open(class_indices_path) as f:
            # {0: "Apple___Apple_scab", 1: ...}
            self.class_indices = {int(k): v for k, v in json.load(f).items()}
        self.num_classes = len(self.class_indices)
        print(f"[Predictor] Ready — {self.num_classes} classes")

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """Resize and normalize a PIL image for MobileNetV2."""
        img = image.convert("RGB").resize((224, 224), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """
        Run inference on a PIL image.
        Returns dict with top prediction + full metadata.
        """
        arr = self.preprocess(image)
        probs = self.model.predict(arr, verbose=0)[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_results = [
            {"class": self.class_indices[i], "confidence": float(probs[i]) * 100}
            for i in top_indices
        ]

        best_class = top_results[0]["class"]
        best_conf  = top_results[0]["confidence"]

        # Split into crop and disease
        parts    = best_class.split("___")
        crop     = parts[0].replace("_", " ") if len(parts) > 0 else best_class
        disease  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        is_healthy = "healthy" in best_class.lower()

        meta = DISEASE_META.get(best_class, DEFAULT_META)

        return {
            "class_raw":        best_class,
            "crop":             crop,
            "disease":          disease,
            "is_healthy":       is_healthy,
            "confidence":       round(best_conf, 1),
            "status":           "Healthy" if is_healthy else "Diseased",
            "severity":         "None" if is_healthy else meta.get("severity_default", "Moderate"),
            "cause":            meta.get("cause", ""),
            "symptoms":         meta.get("symptoms", []),
            "northeast_context": meta.get("northeast_context", DEFAULT_META["northeast_context"]),
            "treatment":        meta.get("treatment", []),
            "organic_options":  meta.get("organic_options", []),
            "prevention":       meta.get("prevention", []),
            "top_predictions":  top_results,
        }


# ─── STANDALONE USAGE ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predictor.py <path/to/image.jpg>")
        sys.exit(1)

    predictor = PlantDiseasePredictor(
        model_path="models/plantdisease_mobilenetv2.h5",
        class_indices_path="models/class_indices.json",
    )
    img = Image.open(sys.argv[1])
    result = predictor.predict(img)

    print("\n" + "="*50)
    print(f"  Crop        : {result['crop']}")
    print(f"  Disease     : {result['disease']}")
    print(f"  Status      : {result['status']}")
    print(f"  Confidence  : {result['confidence']:.1f}%")
    print(f"  Severity    : {result['severity']}")
    print(f"  Cause       : {result['cause']}")
    print("\n  Symptoms:")
    for s in result["symptoms"]:
        print(f"    • {s}")
    print("\n  Treatment:")
    for t in result["treatment"]:
        print(f"    → {t}")
    print("\n  NE India Context:")
    print(f"    {result['northeast_context']}")
    print("="*50)
