import streamlit as st
import torch
import torch.nn as nn
import pickle
from transformers import AutoTokenizer, AutoModel

# --- Configuration & Constants ---
MODEL_NAME = "csebuetnlp/banglabert"
MAX_LEN = 128
# Update these paths to your project folder
MODEL_PATH = r"C:\Users\user\Desktop\PUST_Conf\fastmedbangla_best.pt" 
LABELS_PATH = r"C:\Users\user\Desktop\PUST_Conf\label_encoders.pkl"

# UI Styling
URGENCY_MAP = {
    "Emergency": {"color": "#FF4B4B", "icon": "🚨", "desc": "Immediate Attention Required"},
    "Urgent":    {"color": "#FFA500", "icon": "⚠️", "desc": "Consultation within 24 Hours"},
    "Non-Urgent": {"color": "#21C45D", "icon": "✅", "desc": "Routine Check-up"}
}

 
    

# --- Model Architecture (Matches Training) ---
class FastMedBangla(nn.Module):
    def __init__(self, model_name, n_specialist, n_urgency, n_disease, dropout=0.3):
        super(FastMedBangla, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        
        # Multi-task heads
        self.head_specialist = nn.Linear(hidden_size, n_specialist)
        self.head_urgency = nn.Linear(hidden_size, n_urgency)
        self.head_disease_group = nn.Linear(hidden_size, n_disease)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation for classification
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        
        logits_s = self.head_specialist(pooled_output)
        logits_u = self.head_urgency(pooled_output)
        logits_d = self.head_disease_group(pooled_output)
        
        return logits_s, logits_u, logits_d

# --- Resource Loading (Cached for Speed) ---
@st.cache_resource
def load_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(LABELS_PATH, "rb") as f:
        le = pickle.load(f)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = FastMedBangla(
        MODEL_NAME,
        len(le["specialist"].classes_),
        len(le["urgency"].classes_),
        len(le["disease_group"].classes_)
    )
    
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    
    return model, tokenizer, le, device

# --- Pure Model Inference ---
def run_triage(text, model, tokenizer, le, device):
    inputs = tokenizer(
        text, 
        max_length=MAX_LEN, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        l_s, l_u, l_d = model(inputs["input_ids"], inputs["attention_mask"])

    def process_head(logits, encoder):
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = probs.argmax()
        return encoder.classes_[idx], float(probs[idx]), probs
    
        

    return (
        process_head(l_s, le["specialist"]),
        process_head(l_u, le["urgency"]),
        process_head(l_d, le["disease_group"])
    )

# --- Streamlit Interface ---
st.set_page_config(page_title="FastMedBangla", page_icon="🩺", layout="wide")

st.title("🩺 FastMedBangla: AI Medical Triage")
st.markdown("""
    **Multi-Task BanglaBERT Architecture** for automated patient routing.
    *This system predicts Specialist, Urgency, and Disease Category simultaneously from Bengali natural language symptoms.*
""")

input_text = st.text_area("রোগীর লক্ষণগুলো লিখুন (Enter Symptoms):", placeholder="যেমন: আমার প্রচণ্ড বুকে ব্যথা এবং শ্বাসকষ্ট হচ্ছে...", height=150)

if st.button("🚀 Run AI Analysis", type="primary"):
    if not input_text.strip():
        st.warning("Please enter some symptoms first.")
    else:
        with st.spinner("Analyzing semantic intent..."):
            model, tokenizer, le, device = load_assets()
            (spec, s_conf, s_all), (urg, u_conf, u_all), (dis, d_conf, d_all) = run_triage(input_text, model, tokenizer, le, device)

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👨‍⚕️ Specialist")
            st.metric(label="Department", value=spec, delta=f"{s_conf:.1%} Match")
            st.progress(s_conf)
            with st.expander("Probability Distribution"):
                for name, p in sorted(zip(le["specialist"].classes_, s_all), key=lambda x: -x[1]):
                    st.write(f"{name}: {p:.1%}")

        with col2:
            u_info = URGENCY_MAP.get(urg, {"color": "#999", "icon": "❓", "desc": "Unknown"})
            st.subheader("🕒 Urgency")
            st.markdown(f"""
                <div style="background:{u_info['color']}; padding:20px; border-radius:10px; text-align:center;">
                    <h2 style="color:white; margin:0;">{u_info['icon']} {urg}</h2>
                    <p style="color:white; margin:0; opacity:0.8;">{u_info['desc']}</p>
                </div>
            """, unsafe_allow_html=True)
            st.caption(f"Model Confidence: {u_conf:.1%}")

        with col3:
            st.subheader("📁 Disease Group")
            st.metric(label="Category", value=dis, delta=f"{d_conf:.1%} Match")
            st.progress(d_conf)

        # Scientific Reliability Footer
        if u_conf < 0.40:
            st.info("ℹ️ **Confidence Alert:** The model is uncertain about this case. A human medical professional must review this immediately.")

st.divider()
st.caption("Developed for Research | Built with BanglaBERT Multi-Task Architecture")