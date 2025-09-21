import streamlit as st
import pandas as pd
import os
import subprocess
import plotly.express as px
from supabase import create_client, Client

# Page configuration
st.set_page_config(page_title="Admin Panel", page_icon="🛠️", layout="wide")

st.markdown("""
   <style>
        .stApp {
            background-color: #F0F4F7;
            color: #2E7D32;
        }
    </style>
    """, unsafe_allow_html=True)

# --- Configuration ---
MODEL_FILE = "Effi_WRM.keras"
UPDATED_MODEL_FILE = "Effi_WRM_updated.keras"
IMAGE_DIR = "feedback_images"

# --- Supabase / PostgreSQL Setup ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_ANON_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
FEEDBACK_TABLE = "feedback_records"

# --- Session State Initialization ---
if "password_correct" not in st.session_state:
    st.session_state.password_correct = False
if "confirming_delete" not in st.session_state:
    st.session_state.confirming_delete = None

# --- Authentication ---
def check_password():
    if st.session_state.password_correct:
        return True
    st.header("Admin Login")
    password = st.text_input("Enter Admin Password:", type="password", key="password")
    if st.button("Login"):
        if password == st.secrets["ADMIN_PASSWORD"]:
            st.session_state.password_correct = True
        else:
            st.error("The password you entered is incorrect.")
    return False

if not check_password():
    st.warning("🔒 Please enter the correct password to access the Admin Panel.")
    st.stop()

# --- Main App ---
st.title("🛠️ Waste Classifier Admin Panel")
st.markdown("---")

# Load feedback data from Supabase
response = supabase.table(FEEDBACK_TABLE).select("*").execute()
if response.data is None or len(response.data) == 0:
    st.info("No feedback records yet.")
else:
    df = pd.DataFrame(response.data)
    
    # --- Interactive Dashboard ---
    st.subheader("📊 At a Glance")
    correct_count = df[df['correct'] == 'Yes'].shape[0]
    incorrect_count = df[df['correct'] == 'No'].shape[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Feedbacks", len(df))
    col2.metric("✅ Correct Predictions", correct_count)
    col3.metric("❌ Incorrect Predictions", incorrect_count)
    
    if incorrect_count > 0 or correct_count > 0:
        fig = px.pie(values=[correct_count, incorrect_count], names=['Correct', 'Incorrect'], title='Prediction Accuracy based on Feedback')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # --- Feedback Records Table with Delete ---
    st.subheader("📝 Feedback Records")
    
    for index, row in df.iterrows():
        with st.container():
            if st.session_state.confirming_delete == index:
                st.warning(f"**Are you sure you want to delete this feedback record?** This action cannot be undone.")
                c1, c2, c3 = st.columns([1, 1, 5])
                if c1.button("✅ Yes, Delete", key=f"yes_del_{index}"):
                    # Delete from Supabase
                    supabase.table(FEEDBACK_TABLE).delete().eq("timestamp", row['timestamp']).execute()
                    filename = str(row.get("filename", "")).strip()
                    img_path = os.path.join(IMAGE_DIR, filename)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    st.session_state.confirming_delete = None
                    st.success(f"Record {index} deleted.")
                    st.experimental_rerun()
                if c2.button("❌ Cancel", key=f"no_del_{index}"):
                    st.session_state.confirming_delete = None
                    st.experimental_rerun()
            else:
                cols = st.columns([1, 2, 2, 2, 1])
                filename = str(row.get("filename", "")).strip()
                img_path = os.path.join(IMAGE_DIR, filename)
                
                with cols[0]:
                    if os.path.exists(img_path):
                        st.image(img_path, width=70)
                    else:
                        st.text("No Img")
                
                cols[1].write(f"**Predicted:** `{row.get('predicted', 'N/A')}`")
                
                correct_status = row.get('correct', 'N/A')
                if correct_status == 'Yes' or correct_status == 'हाँ':
                    cols[2].success(f"**Correct?:** {correct_status}")
                else:
                    cols[2].error(f"**Correct?:** {correct_status}")

                cols[3].write(f"**User's Class:** `{row.get('new_class', 'N/A')}`")

                with cols[4]:
                    if st.button("Delete", key=f"del_{index}"):
                        st.session_state.confirming_delete = index
                        st.experimental_rerun()
            
        st.markdown("<hr style='margin:0.5rem 0'>", unsafe_allow_html=True)

    # --- Retrain Section ---
    st.markdown("---")
    st.subheader("⚡ Retrain Model")
    st.write("Retrain the model using the feedback data where the prediction was incorrect.")

    if st.button("Retrain Model with Feedback Data", key="retrain_button"):
        with st.spinner("🚀 Retraining started… Please wait, this may take a few minutes."):
            try:
                result = subprocess.run(
                    ["python", os.path.join(os.getcwd(), "train.py")],
                    capture_output=True, text=True, check=True
                )
                st.success("✅ Training completed successfully!")
                st.text_area("Logs:", result.stdout, height=200)
            except subprocess.CalledProcessError as e:
                st.error("❌ Training script failed.")
                st.text_area("Error:", e.stderr, height=200)
