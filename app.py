# ==========================
# TrashLens - The Ultimate Final Bilingual Version
# ==========================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
import pandas as pd
import random
from PIL import Image
import io 
import base64
import googlemaps 
from datetime import datetime as dt 
import datetime
from streamlit_geolocation import streamlit_geolocation
from geopy.distance import geodesic
import folium
import streamlit.components.v1 as components
import requests

# ==========================
# Page Config 
# ==========================
LOGO_FILE = "logo.jpg"
st.set_page_config(page_title="TrashLens", page_icon=LOGO_FILE, layout="wide")

# ==========================
#  Page CSS
# ==========================

st.markdown("""
    <style>
        .stApp {
            background-color: #F0F4F7; /* App ka background color (halka hara-grey) */
            color: #2E7D32;            /* Text ka color (gehra hara) */
        }
    </style>
    """, unsafe_allow_html=True)

# ==========================
#  File Paths
# ==========================
LOGO_FILE = "logo.jpg"
MODEL_FILE = "Effi_WRM.keras"
UPDATED_MODEL_FILE = "Effi_WRM_updated.keras"
FEEDBACK_FILE = "feedback_records.csv"
FEEDBACK_IMG_DIR = "feedback_images"

# Create feedback directory if it doesn't exist
os.makedirs(FEEDBACK_IMG_DIR, exist_ok=True)

# =====================================================================
# Custom CSS for App-like UI
# =====================================================================
def load_custom_css():
    st.markdown("""
        <style>
            /* ========== Main Font and Colors ========== */
            /* This targets the main app container */
            .st-emotion-cache-1jicfl2 {
                background-color: #F0F4F7; /* App background color */
            }

            /* ========== Title and Header Styles ========== */
            h1, h2, h3 {
                color: #2E7D32 !important; /* Dark green for all headers */
                font-weight: bold !important;
            }

            /* ========== Button Styles ========== */
            .stButton > button {
                border-radius: 25px !important;
                padding: 10px 20px !important;
                font-weight: bold !important;
                border: 2px solid #4CAF50 !important;
                background-color: #4CAF50 !important;
                color: white !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .stButton > button:hover {
                background-color: #45a049 !important;
                border-color: #45a049 !important;
                box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            }
            /* Secondary button style for non-active nav items */
             .stButton > button:not(.st-emotion-cache-19n6bnv) {
                background-color: transparent !important;
                color: #2E7D32 !important;
             }
             .stButton > button:not(.st-emotion-cache-19n6bnv):hover {
                background-color: #E8F5E9 !important;
                color: #2E7D32 !important;
             }

            /* ========== Card Styles for Waste Types ========== */
            .card {
                background-color: #FFFFFF;
                border-radius: 15px;
                padding: 15px;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.2s;
                margin-bottom: 20px;
                height: 350px; /* Fixed height for all cards */
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }
            .card:hover {
                transform: scale(1.03);
            }
            .card img {
                border-radius: 10px;
                max-height: 150px; /* Limit image height */
                object-fit: cover; /* Ensure image covers the area */
                width: 100%;
            }
            .card h3 {
                font-size: 1.2em;
                margin-top: 10px;
                color: #2E7D32;
            }
            /* ======================================================= */
            /* <-- YAHAN SE NAYA CSS CODE PASTE KAR --> */
            /* ======================================================= */
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Tip of the Day Cards */
            .tip-card {
                background-color: #ffffff;
                border-radius: 15px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                border-top: 5px solid;
                margin-bottom: 20px;
            }
            .tip-card-do {
                border-color: #4CAF50;
            }
            .tip-card-dont {
                border-color: #f44336;
            }
            .tip-card .icon {
                font-size: 3rem;
                line-height: 1;
            }
            .tip-card h3 {
                margin-bottom: 10px;
                font-size: 1.5rem;
            }
            .tip-card p {
                font-size: 1.1rem;
                color: #555;
            }

            /* List Items */
            .list-item {
                background-color: #fff;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 10px;
                opacity: 0;
                animation: fadeInUp 0.5s ease-out forwards;
                transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
                border-left: 5px solid;
            }
            .list-item:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.1);
            }
            .list-item-do {
                border-color: #4CAF50;
            }
            .list-item-dont {
                border-color: #f44336;
            }

            /* ======================================================= */
            /* <-- NAYE CSS CODE KA ANT --> */
            /* ======================================================= */    

            /* ========== Top Navigation Bar ========== */
            .st-emotion-cache-1jicfl2 {
                background-color: #FFFFFF;
                border-radius: 30px;
                padding: 10px 5px !important;
                margin-bottom: 2rem !important;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }

        </style>
    """, unsafe_allow_html=True)

# ==========================
#  Model Loading
# ==========================    

@st.cache_resource
def load_my_model():
    model_path = UPDATED_MODEL_FILE if os.path.exists(UPDATED_MODEL_FILE) else MODEL_FILE
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = load_my_model()

    
   # ================================================
# NEW STYLED TOP NAVIGATION BAR (UPDATED VERSION)
# ================================================
def render_top_navbar():
    lang = st.session_state.lang
    page_options_top = {
        "Home": ui_texts['home'][lang],
        "Waste Classifier": ui_texts['classifier'][lang],
        "Waste Types": ui_texts['waste_types'][lang],
        "Do's and Don'ts": ui_texts['dos_donts'][lang]
    }
    cols = st.columns(len(page_options_top))
    for i, (page_key, page_label) in enumerate(page_options_top.items()):
        with cols[i]:
            is_active = (st.session_state.page == page_key)
            button_type = "primary" if is_active else "secondary"
            if st.button(page_label, key=f"top_nav_{page_key}", use_container_width=True, type=button_type):
                if not is_active:
                    st.session_state.page = page_key
                    if page_key != "Waste Classifier":
                        st.session_state.prediction = None
                        st.session_state.uploaded_image = None
                    st.experimental_rerun()
# ==========================
# Internationalization (i18n) Dictionaries
# ==========================
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

ui_texts = {
    "app_title": {"en": "TrashLens", "hi": "ट्रैशलेंस"},
    "go_to": {"en": "Go to", "hi": "इस पेज पर जाएं"},
    "home": {"en": "Home", "hi": "होम"},
    "classifier": {"en": "Waste Classifier", "hi": "अपशिष्ट क्लासिफायर"},
    "waste_types": {"en": "Waste Types", "hi": "अपशिष्ट के प्रकार"},
    "dos_donts": {"en": "Do's and Don'ts", "hi": "क्या करें और क्या न करें"},
    
    # Home Page Content
    "home_title": {"en": "Welcome to TrashLens ♻️", "hi": "ट्रैशलेंस में आपका स्वागत है ♻️"},
    
    "home_subtitle": {
        "en": "Snap->Classify->Recycle- Let AI Guide You.\n\n\nYour smart guide to revolutionizing waste management. We leverage cutting-edge AI to help you classify waste accurately and make sustainable choices effortlessly.",
        "hi": "क्लिक करें -> पहचानें -> रीसायकल करें - AI से मदद लें। \n\n\nअपशिष्ट प्रबंधन में क्रांति लाने के लिए आपका स्मार्ट गाइड। हम आपको कचरे का सही वर्गीकरण करने और सहजता से स्थायी विकल्प चुनने में मदद करने के लिए अत्याधुनिक AI का लाभ उठाते हैं।"
    },
    "home_challenge_title": {"en": "The Challenge: A World Overflowing with Waste", "hi": "चुनौती: कचरे से भरी दुनिया"},
    "home_challenge_text": {
        "en": "Every year, the world generates over 2 billion tonnes of municipal solid waste. A significant portion of this waste ends up in landfills, contributing to pollution. While recycling is a powerful solution, one of the biggest hurdles is improper waste sorting. This is the information gap that TrashLens aims to bridge.",
        "hi": "हर साल, दुनिया 2 अरब टन से अधिक नगरपालिका ठोस अपशिष्ट उत्पन्न करती है। इस कचरे का एक महत्वपूर्ण हिस्सा लैंडफिल में समाप्त हो जाता है, जो प्रदूषण में योगदान देता है। जबकि पुनर्चक्रण एक शक्तिशाली समाधान है, सबसे बड़ी बाधाओं में से एक अनुचित अपशिष्ट छंटाई है। यह सूचना अंतर है जिसे ट्रैशलेंस पाटने का लक्ष्य रखता है।"
    },
    "home_solution_title": {"en": "Our Solution: AI-Powered Clarity", "hi": "हमारा समाधान: AI-संचालित स्पष्टता"},
    "home_solution_text": {
        "en": "TrashLens provides a simple, fast, and accurate solution. Our application is built around a sophisticated computer vision model, **EfficientNetV2**, which has been meticulously trained on thousands of images across 12 distinct waste categories.\n\n1.  **Snap:** You upload an image of a waste item.\n2.  **Classify:** Our AI model analyzes the image in seconds.\n3.  **Act:** The app immediately tells you if the item is recyclable and provides clear steps for proper disposal.",
        "hi": "ट्रैशलेंस इस जटिल समस्या का एक सरल, तेज और सटीक समाधान प्रदान करता है। हमारा एप्लिकेशन एक परिष्कृत कंप्यूटर विज़न मॉडल, **EfficientNetV2** के आसपास बनाया गया है, जिसे 12 अलग-अलग अपशिष्ट श्रेणियों में हजारों छवियों पर सावधानीपूर्वक प्रशिक्षित किया गया है।\n\n1.  **फोटो खींचें:** आप किसी अपशिष्ट वस्तु की एक छवि अपलोड करते हैं।\n2.  **वर्गीकृत करें:** हमारा AI मॉडल सेकंड में छवि का विश्लेषण करता है।\n3.  **कार्यवाही करें:** ऐप तुरंत आपको बताता है कि क्या वस्तु पुनर्चक्रण योग्य है और उचित निपटान के लिए स्पष्ट कदम प्रदान करता है।"
    },
    
    # Do's and Don'ts Content
    "dos_donts_subtitle": {"en": "Follow these simple rules to become a recycling champion!", "hi": "पुनर्चक्रण चैंपियन बनने के लिए इन सरल नियमों का पालन करें!"},
    "dos_header": {"en": "The DOs ✅", "hi": "क्या करें ✅"},
    "donts_header": {"en": "The DON'Ts ❌", "hi": "क्या न करें ❌"},
    "dos_list": {
        "en": ["DO Rinse containers before recycling.", "DO Flatten cardboard boxes.", "DO Check local recycling guidelines.", "DO Separate different types of waste.", "DO Donate usable items like clothes and shoes.", "DO Compost your organic waste.", "DO Use reusable bags for shopping.", "DO Take hazardous waste to special collection sites.", "DO Remove lids from bottles (or check local rules).", "DO Keep paper and cardboard dry."],
        "hi": ["करें पुनर्चक्रण से पहले कंटेनरों को धो लें।", "करें कार्डबोर्ड बक्सों को समतल करें।", "करें स्थानीय पुनर्चक्रण दिशानिर्देशों की जांच करें।", "करें विभिन्न प्रकार के कचरे को अलग करें।", "करें कपड़े और जूते जैसी प्रयोग करने योग्य वस्तुओं को दान करें।", "करें अपने जैविक कचरे की खाद बनाएं।", "करें खरीदारी के लिए पुन: प्रयोज्य बैग का उपयोग करें।", "करें खतरनाक कचरे को विशेष संग्रहण स्थलों पर ले जाएं।", "करें बोतलों से ढक्कन हटा दें (या स्थानीय नियम जांचें)।", "करें कागज और कार्डबोर्ड को सूखा रखें।"]
    },
    "donts_list": {
        "en": ["DON'T put recyclables in a plastic bag.", "DON'T recycle greasy or food-soiled items.", "DON'T 'Wish-cycle' - hoping something is recyclable.", "DON'T throw electronics or batteries in the regular trash.", "DON'T recycle small items like straws.", "DON'T mix different types of glass if not allowed.", "DON'T try to recycle broken glass with regular glass.", "DON'T forget to check the recycling symbol on plastics.", "DON'T leave liquids in bottles or containers.", "DON'T throw away items that can be repaired."],
        "hi": ["न करें पुनर्चक्रण योग्य वस्तुओं को प्लास्टिक की थैली में न डालें।", "न करें चिकनाई युक्त या भोजन से सने सामान का पुनर्चक्रण न करें।", "न करें 'विश-साइकल' - यह उम्मीद न करें कि कुछ पुनर्चक्रण योग्य है।", "न करें इलेक्ट्रॉनिक्स या बैटरियों को साधारण कूड़ेदान में न फेंकें।", "न करें स्ट्रॉ जैसी छोटी वस्तुओं का पुनर्चक्रण न करें।", "न करें यदि अनुमति न हो तो विभिन्न प्रकार के कांच को न मिलाएं।", "न करें टूटे हुए कांच को नियमित कांच के साथ पुनर्चक्रण करने का प्रयास न करें।", "न करें प्लास्टिक पर पुनर्चक्रण प्रतीक की जांच करना न भूलें।", "न करें बोतलों या कंटेनरों में तरल पदार्थ न छोड़ें।", "न करें उन वस्तुओं को न फेंकें जिनकी मरम्मत की जा सकती है।"]
    },
    
    # Other UI Content
    "upload_title": {"en": "♻️ Upload & Classify", "hi": "♻️ छवि अपलोड और वर्गीकृत करें"},
    "result_title": {"en": "🔮 Prediction Result", "hi": "🔮 भविष्यवाणी का परिणाम"},
    "classified_as": {"en": "This Image is Classified as", "hi": "यह छवि इस रूप में वर्गीकृत है"},
    "confidence": {"en": "With the Confidence of", "hi": "इस आत्मविश्वास के "},
    "map_button": {"en": "🗺️ Show Route on Map", "hi": "🗺️ मानचित्र पर मार्ग दिखाएं"},
    "feedback_title": {"en": "📩 Feedback to Improve Model", "hi": "📩 मॉडल को बेहतर बनाने के लिए प्रतिक्रिया"},
    "prediction_correct": {"en": "Is the prediction correct?", "hi": "क्या भविष्यवाणी सही है?"},
    "yes": {"en": "Yes", "hi": "हाँ"},
    "no": {"en": "No", "hi": "नहीं"},
    "select_category": {"en": "Select correct category:", "hi": "सही श्रेणी चुनें:"},
    "submit_feedback": {"en": "Submit Feedback", "hi": "प्रतिक्रिया जमा करें"},
    "classify_another": {"en": "🔙 Classify Another Item", "hi": "🔙 दूसरी वस्तु को वर्गीकृत करें"},
    "prediction_sidebar_title": {"en": "♻️ Prediction Details", "hi": "♻️ भविष्यवाणी का विवरण"},
    "you_uploaded": {"en": "You Uploaded", "hi": "आपने अपलोड किया"},
    "status": {"en": "Status", "hi": "स्थिति"},
    "recyclable": {"en": "Recyclable", "hi": "पुनर्नवीनीकरण योग्य"},
    "not_recyclable": {"en": "Not Recyclable", "hi": "पुनर्नवीनीकरण योग्य नहीं"},
    "disposal_steps": {"en": "See Disposal Steps", "hi": "निपटान के चरण देखें"},
    "feedback_thanks": {"en": "✅ Feedback submitted! Thank you. 💚", "hi": "✅ प्रतिक्रिया जमा हो गई है! धन्यवाद। 💚"}
}

recycling_info = {
    'battery': { "recyclable": True, "steps": {
        "en": ["**1. Do Not Put in Regular Trash:** This is most important. Batteries contain heavy metals like mercury and lead which can leak and contaminate soil and water.", "**2. Find an E-waste Collection Point:** Most cities have special drop-off locations for electronic waste. Search online for 'e-waste collection near me'.", "**3. Tape the Terminals:** For lithium-ion and button cell batteries, it's a good practice to put a small piece of non-conductive tape (like electrical or clear tape) over the ends to prevent any risk of fire."],
        "hi": ["**1. साधारण कूड़ेदान में न डालें:** यह सबसे महत्वपूर्ण है। बैटरियों में पारा और सीसा जैसी भारी धातुएँ होती हैं जो रिसकर मिट्टी और पानी को दूषित कर सकती हैं।", "**2. ई-कचरा संग्रहण केंद्र खोजें:** अधिकांश शहरों में इलेक्ट्रॉनिक कचरे के लिए विशेष ड्रॉप-ऑफ स्थान होते हैं। 'मेरे पास ई-कचरा संग्रहण' के लिए ऑनलाइन खोजें।", "**3. टर्मिनलों पर टेप लगाएं:** लिथियम-आयन और बटन सेल बैटरियों के लिए, आग के किसी भी जोखिम को रोकने के लिए सिरों पर गैर-प्रवाहकीय टेप (जैसे बिजली या स्पष्ट टेप) का एक छोटा टुकड़ा लगाना एक अच्छी आदत है।"]
    }},
    'plastic': { "recyclable": True, "steps": {
        "en": ["**1. Check the Number:** Look for the recycling symbol (a triangle of arrows) and the number inside (1-7). Not all numbers are recyclable everywhere. Check your local municipal guidelines.", "**2. Empty and Rinse:** Make sure the container is completely empty. A quick rinse removes food residue, preventing contamination of other recyclables.", "**3. Lids On or Off?:** Rules vary by city. When in doubt, it's often safer to throw the small plastic lid in the trash.", "**4. Let it Dry:** Shake out excess water. Wet items can damage paper products in a mixed recycling bin."],
        "hi": ["**1. नंबर जांचें:** पुनर्चक्रण प्रतीक (तीरों का त्रिकोण) और अंदर का नंबर (1-7) देखें। जांचें कि आपकी स्थानीय सुविधा उस नंबर को स्वीकार करती है या नहीं।", "**2. खाली और साफ करें:** सुनिश्चित करें कि कंटेनर पूरी तरह से खाली है। किसी भी भोजन या तरल अवशेष को हटाने के लिए इसे पानी से जल्दी से धो लें।", "**3. ढक्कन लगाएं या हटाएं?:** नियम शहर के अनुसार अलग-अलग होते हैं। संदेह होने पर, ढक्कन को कूड़ेदान में फेंक दें।", "**4. सूखने दें:** अतिरिक्त पानी निकाल दें। गीली वस्तुएं मिश्रित रीसाइक्लिंग बिन में कागज उत्पादों को नुकसान पहुंचा सकती हैं।"]
    }},
    'cardboard': { "recyclable": True, "steps": {
        "en": ["**1. Flatten the Box:** Break down and flatten all cardboard boxes. This saves a huge amount of space in recycling bins and trucks.", "**2. Remove Packing Materials:** Take out all plastic bags, bubble wrap, styrofoam, and other materials from inside the box.", "**3. Keep it Dry:** Wet or damp cardboard can't be recycled easily as the fibers are damaged. Keep it away from rain.", "**4. No Food Contamination:** Greasy or food-stained cardboard (like the bottom of a pizza box) cannot be recycled. Tear off the clean parts and trash the greasy sections."],
        "hi": ["**1. बॉक्स को समतल करें:** सभी कार्डबोर्ड बक्सों को तोड़कर समतल करें। इससे रीसाइक्लिंग डिब्बे और ट्रकों में भारी मात्रा में जगह बचती है।", "**2. पैकिंग सामग्री निकालें:** बॉक्स के अंदर से सभी प्लास्टिक बैग, बबल रैप, स्टायरोफोम और अन्य सामग्री निकाल दें।", "**3. इसे सूखा रखें:** गीले या नम कार्डबोर्ड को आसानी से पुनर्नवीनीकरण नहीं किया जा सकता क्योंकि फाइबर क्षतिग्रस्त हो जाते हैं। इसे बारिश से दूर रखें।", "**4. भोजन संदूषण नहीं:** चिकना या भोजन से सना हुआ कार्डबोर्ड (जैसे पिज्जा बॉक्स का निचला भाग) पुनर्नवीनीकरण नहीं किया जा सकता है। साफ हिस्सों को फाड़ दें और चिकना भागों को कूड़ेदान में डाल दें।"]
    }},
    'glass': { "recyclable": True, "steps": {
        "en": ["**1. Empty and Rinse:** Ensure the glass bottle or jar is empty. A quick rinse with water is enough to clean out most residues.", "**2. Remove Lids:** Metal or plastic lids should be removed. They can often be recycled separately.", "**3. Don't Break the Glass:** It's safer for sanitation workers if the glass is intact. Broken glass can also be harder to sort.", "**4. Check Colors:** Some facilities require you to separate glass by color (brown, green, clear). Check your local rules."],
        "hi": ["**1. खाली और साफ करें:** सुनिश्चित करें कि कांच की बोतल या जार खाली है। अधिकांश अवशेषों को साफ करने के लिए पानी से एक त्वरित धुलाई पर्याप्त है।", "**2. ढक्कन हटा दें:** धातु या प्लास्टिक के ढक्कन हटा दिए जाने चाहिए। उन्हें अक्सर अलग से पुनर्नवीनीकरण किया जा सकता है।", "**3. कांच न तोड़ें:** यदि कांच बरकरार है तो यह स्वच्छता कर्मचारियों के लिए सुरक्षित है। टूटे हुए कांच को छांटना भी कठिन हो सकता है।", "**4. रंग जांचें:** कुछ सुविधाओं में आपको कांच को रंग (भूरा, हरा, स्पष्ट) के अनुसार अलग करने की आवश्यकता होती है। अपने स्थानीय नियमों की जांच करें।"]
    }},
    'metal': { "recyclable": True, "steps": {
        "en": ["**1. Empty and Rinse:** For food cans (steel, aluminum), make sure they are empty and rinsed to remove any food.", "**2. Crush if Possible:** Crushing aluminum cans saves a lot of space.", "**3. Aerosol Cans:** Ensure aerosol cans (like deodorant) are completely empty before recycling. Do not puncture or flatten them.", "**4. Labels are OK:** You usually don't need to remove the paper labels from cans."],
        "hi": ["**1. खाली और साफ करें:** खाद्य कैन (स्टील, एल्यूमीनियम) के लिए, सुनिश्चित करें कि वे खाली हैं और किसी भी भोजन को हटाने के लिए धोए गए हैं।", "**2. यदि संभव हो तो कुचलें:** एल्यूमीनियम के डिब्बे कुचलने से बहुत जगह बचती है।", "**3. एरोसोल कैन:** सुनिश्चित करें कि एरोसोल कैन (जैसे डिओडोरेंट) रीसाइक्लिंग से पहले पूरी तरह से खाली हैं। उन्हें पंचर या समतल न करें।", "**4. लेबल ठीक हैं:** आपको आमतौर पर डिब्बे से कागज के लेबल हटाने की आवश्यकता नहीं होती है।"]
    }},
    'biological': { "recyclable": False, "steps": {
        "en": [
            "**1. Separate Your Waste:** Keep a separate small bin in your kitchen for all organic waste like fruit peels, vegetable scraps, and leftover food.",
            "**2. Find a Compost Method:** You can create a compost pile in your backyard, use a compost bin, or find a community composting program.",
            "**3. No Meat or Dairy:** Avoid adding meat, bones, dairy products, or oily foods to your home compost pile as they can attract pests and create bad odors.",
            "**4. Balance greens and Browns:** For good compost, mix 'greens' (like kitchen scraps, grass clippings) with 'browns' (like dried leaves, cardboard, twigs).",
            "**5. Use the Compost:** Once the waste turns into dark, rich soil, you can use it to fertilize your plants, garden, or lawn."
        ],
        "hi": [
            "**1. कचरा अलग करें:** अपनी रसोई में फलों के छिलके, सब्जी के टुकड़े और बचे हुए भोजन जैसे सभी जैविक कचरे के लिए एक अलग छोटा कूड़ेदान रखें।",
            "**2. कम्पोस्ट का तरीका खोजें:** आप अपने आँगन में कम्पोस्ट का ढेर बना सकते हैं, कम्पोस्ट बिन का उपयोग कर सकते हैं, या सामुदायिक कम्पोस्टिंग कार्यक्रम खोज सकते हैं।",
            "**3. मांस या डेयरी नहीं:** अपने घर के कम्पोस्ट में मांस, हड्डियाँ, डेयरी उत्पाद या तैलीय भोजन डालने से बचें क्योंकि वे कीड़ों को आकर्षित कर सकते हैं और दुर्गंध पैदा कर सकते हैं।",
            "**4. हरे और भूरे कचरे को संतुलित करें:** अच्छी खाद के लिए, 'हरे' कचरे (रसोई के स्क्रैप, घास की कतरन) को 'भूरे' कचरे (सूखे पत्ते, कार्डबोर्ड, टहनियाँ) के साथ मिलाएं।",
            "**5. खाद का उपयोग करें:** जब कचरा गहरी, पोषक मिट्टी में बदल जाए, तो आप इसका उपयोग अपने पौधों, बगीचे या लॉन में खाद डालने के लिए कर सकते हैं।"
        ]
    }},
    'paper': { "recyclable": True, "steps": {
        "en": ["**1. Keep it Clean and Dry:** Only clean paper can be recycled. Stained paper (food, grease, paint) should be thrown away.", "**2. Remove Attachments:** Remove plastic wrappers, spiral bindings, and large metal clips.", "**3. No Shredded Paper in Mixed Bins:** Loose shredded paper can jam sorting machinery. Put it in a sealed paper bag and label it 'shredded paper', or check if your local facility accepts it."],
        "hi": ["**1. इसे साफ और सूखा रखें:** केवल साफ कागज को ही पुनर्नवीनीकरण किया जा सकता है। सना हुआ कागज (भोजन, ग्रीस, पेंट) फेंक दिया जाना चाहिए।", "**2. संलग्नक निकालें:** प्लास्टिक रैपर, सर्पिल बाइंडिंग और बड़े धातु क्लिप निकालें।", "**3. मिश्रित डिब्बे में कटा हुआ कागज नहीं:** ढीला कटा हुआ कागज छँटाई मशीनरी को जाम कर सकता है। इसे एक सीलबंद कागज की थैली में रखें और इसे 'कटा हुआ कागज' के रूप में लेबल करें, या जांचें कि क्या आपकी स्थानीय सुविधा इसे स्वीकार करती है।"]
    }},
    
    # --- YAHAN SE NAYI ENTRIES ADD HUI HAIN ---
    'shoes': { "recyclable": True, "steps": {
        "en": ["**1. Assess Condition:** Are the shoes still wearable? If yes, donation is the best option.", "**2. Donate if Usable:** Give them to a local charity, thrift store, or a shoe donation program. This extends the product's life.", "**3. Find a Recycling Program:** For worn-out shoes, many brands (like Nike, Adidas) have take-back programs to recycle them into new products or playground surfaces.", "**4. Separate Parts if Possible:** For some recycling, separating the rubber sole from the fabric upper can be helpful, but this is not usually required.", "**5. Clean Before Donating:** If you are donating, please give them a quick clean as a courtesy."],
        "hi": ["**1. स्थिति का आकलन करें:** क्या जूते अभी भी पहनने योग्य हैं? यदि हाँ, तो दान सबसे अच्छा विकल्प है।", "**2. प्रयोग करने योग्य होने पर दान करें:** उन्हें एक स्थानीय चैरिटी, थ्रिफ्ट स्टोर, या जूता दान कार्यक्रम में दें। यह उत्पाद के जीवन का विस्तार करता है।", "**3. एक पुनर्चक्रण कार्यक्रम खोजें:** घिसे-पिटे जूतों के लिए, कई ब्रांडों (जैसे Nike, Adidas) के पास उन्हें नए उत्पादों या खेल के मैदान की सतहों में पुनर्चक्रण करने के लिए टेक-बैक कार्यक्रम होते हैं।", "**4. यदि संभव हो तो भागों को अलग करें:** कुछ पुनर्चक्रण के लिए, रबर के एकमात्र को कपड़े के ऊपरी हिस्से से अलग करना सहायक हो सकता है, लेकिन यह आमतौर पर आवश्यक नहीं है।", "**5. दान करने से पहले साफ करें:** यदि आप दान कर रहे हैं, तो कृपया शिष्टाचार के तौर पर उन्हें एक त्वरित सफाई दें।"]
    }},
    'clothes': { "recyclable": True, "steps": {
        "en": ["**1. Donate First:** The best form of recycling is reuse. If clothes are in good condition (no major tears or stains), donate them to a local charity.", "**2. Textile Recycling Bins:** Look for textile recycling bins in your community for clothes that are too worn to be donated.", "**3. Repurpose at Home:** Old t-shirts and towels make excellent cleaning rags, saving you money.", "**4. Check with Animal Shelters:** Many animal shelters accept old towels, blankets, and sheets for animal bedding.", "**5. H&M or Zara Programs:** Some major clothing retailers have in-store collection programs that accept any clothing from any brand, which they then recycle."],
        "hi": ["**1. पहले दान करें:** पुनर्चक्रण का सबसे अच्छा रूप पुन: उपयोग है। यदि कपड़े अच्छी स्थिति में हैं (कोई बड़ी खराबी या दाग नहीं), तो उन्हें एक स्थानीय चैरिटी में दान करें।", "**2. कपड़ा पुनर्चक्रण डिब्बे:** उन कपड़ों के लिए अपने समुदाय में कपड़ा पुनर्चक्रण डिब्बे देखें जो दान करने के लिए बहुत घिसे-पिटे हैं।", "**3. घर पर पुन: उपयोग करें:** पुरानी टी-शर्ट और तौलिये उत्कृष्ट सफाई के कपड़े बनाते हैं, जिससे आपके पैसे बचते हैं।", "**4. पशु आश्रयों से संपर्क करें:** कई पशु आश्रय जानवरों के बिस्तर के लिए पुराने तौलिये, कंबल और चादरें स्वीकार करते हैं।", "**5. H&M या Zara कार्यक्रम:** कुछ प्रमुख कपड़ा खुदरा विक्रेताओं के पास इन-स्टोर संग्रह कार्यक्रम होते हैं जो किसी भी ब्रांड के किसी भी कपड़े को स्वीकार करते हैं, जिसे वे फिर से रीसायकल करते हैं।"]
    }},
    'trash': { "recyclable": False, "steps": {
        "en": ["**1. Confirm it's Trash:** This category is for items that truly cannot be recycled. This includes items like chip bags, dirty diapers, broken ceramics, and styrofoam.", "**2. Bag it Securely:** To keep your bin clean and prevent litter, please place all trash into a sealed trash bag.", "**3. General Waste Bin:** Dispose of the bag in your designated general waste or landfill bin.", "**4. Hazardous Waste is Different:** Do not put hazardous items like paint, chemicals, or medical waste in the regular trash. They need special disposal.", "**5. Reduce First:** The best way to manage trash is to create less of it. Try to choose products with less packaging or reusable alternatives."],
        "hi": ["**1. पुष्टि करें कि यह कचरा है:** यह श्रेणी उन वस्तुओं के लिए है जिन्हें वास्तव में पुनर्नवीनीकरण नहीं किया जा सकता है। इसमें चिप्स के बैग, गंदे डायपर, टूटे हुए सिरेमिक और स्टायरोफोम जैसी वस्तुएं शामिल हैं।", "**2. इसे सुरक्षित रूप से बैग में डालें:** अपने बिन को साफ रखने और कूड़े को फैलने से रोकने के लिए, कृपया सभी कचरे को एक सीलबंद कचरा बैग में रखें।", "**3. सामान्य अपशिष्ट बिन:** बैग को अपने निर्दिष्ट सामान्य अपशिष्ट या लैंडफिल बिन में निपटाएं।", "**4. खतरनाक अपशिष्ट अलग है:** पेंट, रसायन, या चिकित्सा अपशिष्ट जैसी खतरनाक वस्तुओं को नियमित कचरे में न डालें। उन्हें विशेष निपटान की आवश्यकता होती है।", "**5. पहले कम करें:** कचरे का प्रबंधन करने का सबसे अच्छा तरीका इसे कम बनाना है। कम पैकेजिंग या पुन: प्रयोज्य विकल्पों वाले उत्पादों को चुनने का प्रयास करें।"]
    }},
}

# Make sure all categories are in the dictionary
for cat in class_names:
    if cat in ['brown-glass', 'green-glass', 'white-glass']:
        recycling_info[cat] = recycling_info['glass']
    elif cat not in recycling_info:
        recycling_info[cat] = { "recyclable": False, "steps": {"en": ["This item is generally considered non-recyclable. Please dispose of it in the regular trash bin."], "hi": ["यह वस्तु आम तौर पर गैर-पुनर्नवीनीकरण योग्य मानी जाती है। कृपया इसे नियमित कूड़ेदान में फेंक दें।"]} }
waste_info_details = {
    "battery": {
        "image_url": "images/battery.jpg",
        "title": {"en": "Battery", "hi": "बैटरी"},
        "info": {
            "en": "Includes AA, AAA, and car batteries. Requires special disposal at e-waste facilities.",
            "hi": "इसमें AA, AAA, और कार बैटरी शामिल हैं। ई-कचरा सुविधाओं में विशेष निपटान की आवश्यकता है।"
        }
    },
    "biological": {
        "image_url": "images/biological.jpg",
        "title": {"en": "Biological", "hi": "जैविक"},
        "info": {
            "en": "Food scraps, fruit peels, yard waste. Can be composted to create nutrient-rich soil.",
            "hi": "भोजन के स्क्रैप, फलों के छिलके, यार्ड का कचरा। पोषक तत्वों से भरपूर मिट्टी बनाने के लिए खाद बनाई जा सकती है।"
        }
    },
    "brown-glass": {
        "image_url": "images/brown-glass.jpg",
        "title": {"en": "Brown Glass", "hi": "भूरा कांच"},
        "info": {
            "en": "Beer bottles, medicine bottles. Glass is 100% recyclable. Must be rinsed before recycling.",
            "hi": "बियर की बोतलें, दवा की बोतलें। कांच 100% पुनर्नवीनीकरण योग्य है। पुनर्चक्रण से पहले धोना चाहिए।"
        }
    },
    "cardboard": {
        "image_url": "images/cardboard.jpg",
        "title": {"en": "Cardboard", "hi": "कार्डबोर्ड"},
        "info": {
            "en": "Packaging boxes, cartons. Should be flattened to save space. Must be clean and dry.",
            "hi": "पैकेजिंग बक्से, डिब्बों। जगह बचाने के लिए समतल किया जाना चाहिए। साफ और सूखा होना चाहिए।"
        }
    },
    "clothes": {
        "image_url": "images/clothes.jpg",
        "title": {"en": "Clothes", "hi": "कपड़े"},
        "info": {
            "en": "Unwanted garments, textiles. Can be donated if in good condition or recycled into new fibers.",
            "hi": "अवांछित वस्त्र, कपड़े। अच्छी स्थिति में होने पर दान किया जा सकता है या नए रेशों में पुनर्नवीनीकरण किया जा सकता है।"
        }
    },
    "green-glass": {
        "image_url": "images/green-glass.jpg",
        "title": {"en": "green Glass", "hi": "हरा कांच"},
        "info": {
            "en": "Wine bottles, juice bottles. Fully recyclable. Sorting glass by color is important.",
            "hi": "शराब की बोतलें, जूस की बोतलें। पूरी तरह से पुनर्नवीनीकरण योग्य। कांच को रंग के अनुसार छांटना महत्वपूर्ण है।"
        }
    },
    "metal": {
        "image_url": "images/metal.jpg",
        "title": {"en": "Metal", "hi": "धातु"},
        "info": {
            "en": "Aluminum cans, steel food cans. Highly recyclable. Rinsing is necessary to remove food residue.",
            "hi": "एल्यूमीनियम के डिब्बे, स्टील के खाद्य डिब्बे। अत्यधिक पुनर्नवीनीकरण योग्य। भोजन के अवशेषों को हटाने के लिए धोना आवश्यक है।"
        }
    },
    "paper": {
        "image_url": "images/paper.jpg",
        "title": {"en": "Paper", "hi": "कागज"},
        "info": {
            "en": "Newspapers, magazines, office paper. Must be kept clean and dry for effective recycling.",
            "hi": "समाचार पत्र, पत्रिकाएं, कार्यालय का कागज। प्रभावी पुनर्चक्रण के लिए साफ और सूखा रखना चाहिए।"
        }
    },
    "plastic": {
        "image_url": "images/plastic.jpg",
        "title": {"en": "Plastic", "hi": "प्लास्टिक"},
        "info": {
            "en": "Bottles, containers, bags. Check the recycling symbol (1-7) to know the type. Should be rinsed.",
            "hi": "बोतलें, कंटेनर, बैग। प्रकार जानने के लिए पुनर्चक्रण प्रतीक (1-7) की जांच करें। धोया जाना चाहिए।"
        }
    },
    "shoes": {
        "image_url": "images/shoes.jpg",
        "title": {"en": "Shoes", "hi": "जूते"},
        "info": {
            "en": "All types of footwear. Can be donated if wearable. Some programs recycle them into new materials.",
            "hi": "सभी प्रकार के जूते। पहनने योग्य होने पर दान किया जा सकता है। कुछ कार्यक्रम उन्हें नई सामग्रियों में पुनर्नवीनीकरण करते हैं।"
        }
    },
    "trash": {
        "image_url": "images/trash.jpg",
        "title": {"en": "Trash", "hi": "कचरा"},
        "info": {
            "en": "General, non-recyclable waste. Includes items like chip bags, styrofoam, and mixed-material products.",
            "hi": "सामान्य, गैर-पुनर्नवीनीकरण योग्य कचरा। इसमें चिप्स के बैग, स्टायरोफोम और मिश्रित-सामग्री वाले उत्पाद शामिल हैं।"
        }
    },
    "white-glass": {
        "image_url": "images/white-glass.jpg",
        "title": {"en": "White Glass", "hi": "सफेद कांच"},
        "info": {
            "en": "Clear glass jars (jam, pickles), beverage bottles. Fully recyclable but should be separated from colored glass.",
            "hi": "साफ कांच के जार (जैम, अचार), पेय की बोतलें। पूरी तरह से पुनर्नवीनीकरण योग्य लेकिन रंगीन कांच से अलग किया जाना चाहिए।"
        }
    }
}

# ==========================
# Session State & Helper Functions
# ==========================
if 'page' not in st.session_state: st.session_state.page = "Home"
if 'prediction' not in st.session_state: st.session_state.prediction = None
if 'uploaded_image' not in st.session_state: st.session_state.uploaded_image = None
if 'lang' not in st.session_state: st.session_state.lang = 'en'
if 'show_map' not in st.session_state: st.session_state.show_map = False
if 'feedback_submitted' not in st.session_state: st.session_state.feedback_submitted = False
if 'user_location' not in st.session_state: st.session_state.user_location = None

def classify_image(img):
    img_resized = img.resize((384, 384)); img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0); img_array = preprocess_input(img_array)
    preds = model.predict(img_array); pred_class_index = np.argmax(preds, axis=1)[0]
    pred_class_name = class_names[pred_class_index]; confidence = preds[0][pred_class_index] * 100
    return pred_class_name, confidence, preds[0]
def get_route(start_lon, start_lat, end_lon, end_lat):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        r = requests.get(url); r.raise_for_status(); route = r.json()['routes'][0]['geometry']['coordinates']
        route = [(coord[1], coord[0]) for coord in route]; return route
    except Exception: return None
def save_feedback(predicted, correct, new_class, pil_img):
    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame(columns=["timestamp", "filename", "predicted", "correct", "new_class"]).to_csv(FEEDBACK_FILE, index=False)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.png"
    filepath = os.path.join(FEEDBACK_IMG_DIR, filename)
    pil_img.save(filepath)
    
    df = pd.read_csv(FEEDBACK_FILE)
    new_record = {
        "timestamp": timestamp, "filename": filename, "predicted": predicted,
        "correct": correct, "new_class": new_class
    }
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    df.to_csv(FEEDBACK_FILE, index=False)
# PURANE FUNCTION KO HATAKAR YE NAYA WALA PASTE KAR

@st.cache_data
def get_image_as_base64(path, max_size=(200, 200)):
    """
    Reads an image, resizes it, and converts to a Base64 encoded string.
    """
    try:
        # Image ko Pillow se kholo
        img = Image.open(path)
        
        # Image ka size chota karo
        img.thumbnail(max_size)
        
        # Image ko memory mein save karo
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        
        # Ab is choti image ko Base64 mein badlo
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None


# ==========================
# Page Rendering Functions
# ==========================
def render_home_page():
    lang = st.session_state.lang
    st.title(ui_texts['home_title'][lang])
    col1, col2, col3 = st.columns([1, 2, 1]); 
    with col2: st.image(LOGO_FILE, width=400)
    st.markdown(f"<p style='text-align: center; font-size: 20px;'>{ui_texts['home_subtitle'][lang]}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader(ui_texts['home_challenge_title'][lang])
    st.write(ui_texts['home_challenge_text'][lang])
    st.subheader(ui_texts['home_solution_title'][lang])
    st.write(ui_texts['home_solution_text'][lang])

def render_classifier_page():
    lang = st.session_state.lang
    st.title(ui_texts['classifier'][lang])
    if st.session_state.prediction is None:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader")
        if uploaded_file:
            img = Image.open(uploaded_file).convert('RGB')
            st.session_state.uploaded_image = img
            with st.spinner('🧠 AI is thinking...'):
                pred_class, conf, preds = classify_image(img)
            st.session_state.prediction = pred_class; st.session_state.confidence = conf; st.session_state.preds = preds
            st.session_state.feedback_submitted = False; st.experimental_rerun()
    else:
        st.header(ui_texts['result_title'][lang])
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(st.session_state.uploaded_image, caption=ui_texts['you_uploaded'][lang], width=150)
        with col2:
            st.markdown(f"### {ui_texts['classified_as'][lang]}: **{st.session_state.prediction.capitalize()}**")
            st.metric(label=ui_texts['confidence'][lang], value=f"{st.session_state.confidence:.2f}%")
            # ===========================================================
        # --- नया कोड यहाँ से शुरू ---
        # ===========================================================
        st.markdown("---") # एक लाइन खींचेगा
        
        pred_class = st.session_state.prediction
        info = recycling_info.get(pred_class, {})
        
        # स्टेटस (Recyclable है या नहीं) दिखाएगा
        status_text = ui_texts['recyclable'][lang] if info.get('recyclable') else ui_texts['not_recyclable'][lang]
        if info.get('recyclable'):
            st.success(f"**{ui_texts['status'][lang]}:** {status_text} ♻️")
        else:
            st.error(f"**{ui_texts['status'][lang]}:** {status_text} 🗑️")

        # निपटान के स्टेप्स (Disposal Steps) एक expander में दिखाएगा
        with st.expander(ui_texts['disposal_steps'][lang]):
            for step in info.get('steps', {}).get(lang, []):
                st.markdown(f"{step}")
        # ===========================================================
        # --- नए कोड का अंत ---
        # ===========================================================
        st.markdown("---")
        st.subheader("📍 Find Nearest Disposal Center")
        location = streamlit_geolocation()
        if location and location.get('latitude'):
            st.session_state.user_location = location
            st.info("Your location has been found. Click the button below to see the route on a map.")
            if st.button(ui_texts['map_button'][lang]):
                st.session_state.show_map = True; st.experimental_rerun()
        else:
            st.warning("To use the map feature, please **'Allow'** location access in your browser.")
        st.markdown("---")
        st.subheader(ui_texts['feedback_title'][lang])
        if not st.session_state.feedback_submitted:
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                correct = st.radio(ui_texts['prediction_correct'][lang], [ui_texts['yes'][lang], ui_texts['no'][lang]], index=0, key="feedback_radio")
            new_class = None
            if correct == ui_texts['no'][lang]:
                with f_col2:
                    new_class = st.selectbox(ui_texts['select_category'][lang], class_names, key="feedback_selectbox")
            if st.button(ui_texts['submit_feedback'][lang]):
                save_feedback(st.session_state.prediction, correct, new_class, st.session_state.uploaded_image)
                st.session_state.feedback_submitted = True; st.success("✅ Feedback submitted! Thank you."); st.experimental_rerun()
        else:
            st.info(ui_texts['feedback_thanks'][lang])
        if st.button(ui_texts['classify_another'][lang]):
            st.session_state.prediction = None; st.session_state.uploaded_image = None; st.experimental_rerun()

# PURANE FUNCTION KO HATAKAR YE POORA NAYA FUNCTION DAAL DE

def render_map_page():
    # Step 1: CSS Styling
    CUSTOM_CSS = """
<style>
    h1 { text-align: center; color: #2E7D32; }
    .distance-text { text-align: center; font-size: 1.2em; font-weight: bold; margin-bottom: 20px; }
    
    .custom-button {
        display: block; /* ताकि पूरी चौड़ाई ले सके */
        border-radius: 25px !important; /* गोल किनारे */
        padding: 10px 20px !important; /* अंदर की पैडिंग */
        font-weight: bold !important; /* टेक्स्ट बोल्ड */
        border: 2px solid #4CAF50 !important; /* हरा बॉर्डर */
        background-color: #F0F4F7 !important; /* हल्का बैकग्राउंड, इमेज के जैसा */
        color: #2E7D32 !important; /* गहरा हरा टेक्स्ट */
        transition: all 0.2s ease-in-out !important; /* स्मूथ ट्रांजीशन */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* हल्का शैडो */
        text-decoration: none !important; /* लिंक के नीचे अंडरलाइन हटा दें */
        width: fit-content; /* टेक्स्ट जितना चौड़ा हो */
        margin: 0 auto; /* सेंटर में लाने के लिए */
    }
    .custom-button:hover {
        background-color: #C8E6C9 !important; /* होवर पर थोड़ा गहरा बैकग्राउंड */
        border-color: #388E3C !important; /* होवर पर गहरा हरा बॉर्डर */
        box-shadow: 0 6px 8px rgba(0,0,0,0.15); /* होवर पर थोड़ा गहरा शैडो */
        color: #2E7D32 !important; /* टेक्स्ट का रंग वही रहेगा */
    }
</style>
"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🗺️ Route to Nearest Disposal Center")
    if st.button("⬅️ Go back to Classifier"):
        st.session_state.show_map = False
        st.session_state.prediction = None # Prediction reset kar do taaki nayi image upload ho sake
        st.experimental_rerun()
        return
    user_loc_data = st.session_state.get('user_location')

    if not user_loc_data or not user_loc_data.get('latitude'):
        st.error("User location not found. Please return to the classifier page.")

    user_lat, user_lon = user_loc_data['latitude'], user_loc_data['longitude']
    user_loc_tuple = (user_lat, user_lon)

    try:
        waste_centers_df = pd.read_csv("centers.csv")
    except FileNotFoundError:
        st.error("Error: `centers.csv` file not found.")
        return

    waste_centers_df['distance'] = waste_centers_df.apply(lambda row: geodesic(user_loc_tuple, (row['lat'], row['lon'])).km, axis=1)
    nearest_center = waste_centers_df.loc[waste_centers_df['distance'].idxmin()]
    
    st.success(f"✅ Nearest Center: **{nearest_center['Name']}** (~{nearest_center['distance']:.2f} km away)")

    # =============================================================
    # YAHAN SE ASLI JAADU SHURU HOTA HAI (IF-ELSE BLOCK)
    # =============================================================

    # Pehle check karo ki 'secrets.toml' file mein Google API key hai ya nahi
    if "google_maps_api_key" in st.secrets and st.secrets["google_maps_api_key"]:
        st.info("🚀 Using Google Maps for navigation.")
        with st.spinner("Generating route with Google Maps..."):
            try:
                # Google Maps ka istemaal karke route dikhao
                gmaps = googlemaps.Client(key=st.secrets["google_maps_api_key"])
                
                # Directions ka request bhejo
                directions_result = gmaps.directions((user_lat, user_lon),
                                                     (nearest_center['lat'], nearest_center['lon']),
                                                     mode="driving",
                                                     departure_time=dt.now())
                
                # Google Maps ka embed URL banao
                embed_url = f"https://www.google.com/maps/embed/v1/directions?key={st.secrets['google_maps_api_key']}&origin={user_lat},{user_lon}&destination={nearest_center['lat']},{nearest_center['lon']}&mode=driving"
                
                # Iframe ka use karke map dikhao
                components.html(f'<iframe width="100%" height="600" frameborder="0" style="border:0" src="{embed_url}" allowfullscreen></iframe>', height=620)

            except Exception as e:
                st.error(f"Could not load Google Maps. Error: {e}")
                st.warning("Falling back to the default map.")
                # Agar Google Maps fail ho jaaye to OSRM use karo (ye line zaroori nahi hai, par achhi practice hai)

    else:
        # Agar Google API Key nahi hai to Folium wala map chalega
        st.info("ℹ️ Google Maps API key not found. Using default map.")
        with st.spinner("Generating route..."):
            route = get_route(user_lon, user_lat, nearest_center['lon'], nearest_center['lat'])
            m = folium.Map(
            location=[user_lat, user_lon], 
            zoom_start=12, 
            tiles='http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', 
            attr='Google'
        )

            folium.Marker(
                user_loc_tuple, popup="Your Location",
                icon=folium.Icon(color="blue", icon="user", prefix="fa")
            ).add_to(m)

            for idx, row in waste_centers_df.iterrows():
                is_nearest = (row['Name'] == nearest_center['Name'])
                folium.Marker(
                    [row['lat'], row['lon']],
                    popup=f"<b>{row['Name']}</b><br>~{row['distance']/1000:.2f} km away",
                    icon=folium.Icon(color="green" if is_nearest else "gray", icon="recycle", prefix="fa")
                ).add_to(m)

            if route:
                folium.PolyLine(route, weight=5, color='#006400', opacity=0.8).add_to(m)
                m.fit_bounds(folium.PolyLine(route).get_bounds())
            
            components.html(m._repr_html_(), height=900)

        # Step 6: Navigation Button (Dono maps ke neeche dikhega)
        gmaps_url = f"https://www.google.com/maps/dir/?api=1&destination={nearest_center['lat']},{nearest_center['lon']}"
        st.markdown(f'<a href="{gmaps_url}" target="_blank" class="custom-button">Navigate to Nearest</a>', unsafe_allow_html=True)   

# PURANE FUNCTION KI JAGAH YE NAYA FUNCTION DAAL DE

def render_waste_types_page():
    lang = st.session_state.lang
    st.title(ui_texts['waste_types'][lang])
    st.write({
        "en": "Learn to identify different categories of waste for better sorting and recycling.",
        "hi": "बेहतर छंटाई और पुनर्चक्रण के लिए कचरे की विभिन्न श्रेणियों को पहचानना सीखें।"
    }[lang])
    st.markdown("---")

    # Create columns for the card layout
    cols = st.columns(3) # 4 cards per row
    col_index = 0

    for waste_type, details in waste_info_details.items():
        with cols[col_index]:
            # <-- YAHAN IMAGE KO BASE64 MEIN CONVERT KIYA
            image_path = details['image_url']
            image_base64 = get_image_as_base64(image_path, max_size=(200, 200))
            
            # Use markdown to create a styled card
            if image_base64: # Check karo ki image load hui ya nahi
                card_html = f"""
                    <div class="card">
                        <div>
                            <img src="data:image/jpeg;base64,{image_base64}" alt="{details['title'][lang]}">
                            <h3>{details['title'][lang]}</h3>
                            <p>{details['info'][lang]}</p>
                        </div>
                    </div>
                """
                # <-- UPAR DEKH:
                # 1. <img src="..."> mein naya base64 variable use kiya.
                # 2. <p> tag mein info text add kiya.
                
                st.markdown(card_html, unsafe_allow_html=True)
        
        col_index = (col_index + 1) % 3 # Move to the next column

# PURANE FUNCTION KI JAGAH YE NAYA WALA DAAL DE

# PURANE FUNCTION KI JAGAH YE NAYA WALA DAAL DE

def render_dos_donts_page():
    lang = st.session_state.lang
    st.title(ui_texts['dos_donts'][lang])
    st.write(ui_texts['dos_donts_subtitle'][lang])
    st.markdown("---")

    # --- Tip of the Day ---
    st.header("✨ Tip of the Day")
    
    # Ek random Do aur Don't chuno
    random_do = random.choice(ui_texts['dos_list'][lang])
    random_dont = random.choice(ui_texts['donts_list'][lang])
    
    col1, col2 = st.columns(2)
    with col1:
        do_card_html = f"""
            <div class="tip-card tip-card-do">
                <div class="icon">✅</div>
                <h3>DO</h3>
                <p>{random_do.replace('DO', '')}</p>
            </div>
        """
        st.markdown(do_card_html, unsafe_allow_html=True)
    
    with col2:
        dont_card_html = f"""
            <div class="tip-card tip-card-dont">
                <div class="icon">❌</div>
                <h3>DON'T</h3>
                <p>{random_dont.replace("DON'T", '')}</p>
            </div>
        """
        st.markdown(dont_card_html, unsafe_allow_html=True)

    st.markdown("---")
    st.header("All Tips")
    
    # --- Poori List ---
    col1, col2 = st.columns(2)
    with col1:
        # Yahan hum apne naye 'list-item' ka use kar rahe hain
        for i, item in enumerate(ui_texts['dos_list'][lang]):
            item_html = f'<div class="list-item list-item-do" style="animation-delay: {i * 0.5}s;">{item}</div>'
            st.markdown(item_html, unsafe_allow_html=True)
            
    with col2:
        # Yahan hum apne naye 'list-item' ka use kar rahe hain
        for i, item in enumerate(ui_texts['donts_list'][lang]):
            item_html = f'<div class="list-item list-item-dont" style="animation-delay: {i * 0.5}s;">{item}</div>'
            st.markdown(item_html, unsafe_allow_html=True)

# ==========================
# Main App Logic
# ==========================

load_custom_css()
lang = st.session_state.lang
with st.sidebar:
    if st.session_state.prediction is not None:
        st.header(ui_texts['prediction_sidebar_title'][lang])
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption=ui_texts['you_uploaded'][lang], width=150)
        pred_class = st.session_state.prediction
        info = recycling_info.get(pred_class, {})
        status_text = ui_texts['recyclable'][lang] if info.get('recyclable') else ui_texts['not_recyclable'][lang]
        if info.get('recyclable'): st.success(f"**{ui_texts['status'][lang]}:** {status_text}")
        else: st.error(f"**{ui_texts['status'][lang]}:** {status_text}")
        with st.expander(ui_texts['disposal_steps'][lang]):
            for step in info.get('steps', {}).get(lang, []):
                st.markdown(f"{step}")
    else:
        st.title(ui_texts['app_title'][lang])
        st.image(LOGO_FILE, width=150)
        st.markdown("---")
        lang_choice = st.selectbox("Language / भाषा", ["English", "हिन्दी"], index=0 if lang == 'en' else 1)
        if (lang_choice == "हिन्दी" and lang == 'en') or (lang_choice == "English" and lang == 'hi'):
            st.session_state.lang = 'hi' if lang_choice == "हिन्दी" else 'en'
            st.experimental_rerun()
        
        page_options = { "Home": ui_texts['home'][lang], "Waste Classifier": ui_texts['classifier'][lang], "Waste Types": ui_texts['waste_types'][lang], "Do's and Don'ts": ui_texts['dos_donts'][lang] }
        current_page_label = page_options.get(st.session_state.page)
        current_index = list(page_options.values()).index(current_page_label) if current_page_label in page_options.values() else 0
        selected_label = st.radio(ui_texts['go_to'][lang], list(page_options.values()), key="navigation", index=current_index)
        for key, value in page_options.items():
            if value == selected_label and st.session_state.page != key:
                st.session_state.page = key
                if st.session_state.page != "Waste Classifier":
                    st.session_state.prediction = None; st.session_state.uploaded_image = None
                st.experimental_rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 TrashLens")

# --- Page Routing ---

# ADD THIS ONE LINE HERE
render_top_navbar() 
if st.session_state.get('show_map', False):
    render_map_page()
else:
    page = st.session_state.get('page', 'Home')
    if page == "Home": render_home_page()
    elif page == "Waste Classifier": render_classifier_page()
    elif page == "Waste Types": render_waste_types_page()
    elif page == "Do's and Don'ts": render_dos_donts_page()