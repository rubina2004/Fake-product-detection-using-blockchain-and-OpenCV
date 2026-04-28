import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sqlite3
import hashlib
import qrcode
from datetime import datetime
import pandas as pd
import plotly.express as px

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    try:
        return MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    except:
        return None

dl_model = load_model()

def extract_features(file):
    img = Image.open(file).resize((224,224)).convert("RGB")
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)
    return dl_model.predict(arr, verbose=0)

def deep_similarity(img_bytes, file):
    if dl_model is None:
        return 0.0
    f1 = extract_features(io.BytesIO(img_bytes))
    f2 = extract_features(file)
    return float(cosine_similarity(f1,f2)[0][0]*100)

# ---------------- DATABASE ----------------
conn = sqlite3.connect("products.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""CREATE TABLE IF NOT EXISTS products(
id TEXT PRIMARY KEY,name TEXT,manufacturer TEXT,
batch TEXT,date TEXT,image BLOB,hash TEXT,prev_hash TEXT)""")

cursor.execute("""CREATE TABLE IF NOT EXISTS history(
id INTEGER PRIMARY KEY AUTOINCREMENT,
product_id TEXT,result TEXT,score REAL,timestamp TEXT)""")

conn.commit()

# ---------------- FUNCTIONS ----------------
def generate_hash(data):
    return hashlib.sha256(data).hexdigest()

def generate_qr(data):
    qr = qrcode.make(data)
    buf = io.BytesIO()
    qr.save(buf)
    return buf.getvalue()

def generate_product_id():
    cursor.execute("SELECT COUNT(*) FROM products")
    return f"PD{cursor.fetchone()[0]+1:03d}"

def orb_similarity(img_bytes,file):
    img1 = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    img2 = np.array(Image.open(file).convert("RGB"))

    orb=cv2.ORB_create()
    kp1,des1=orb.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY),None)
    kp2,des2=orb.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),None)

    if des1 is None or des2 is None:
        return 0.0

    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1,des2,k=2)
    good=[m for m,n in matches if m.distance<0.75*n.distance]

    return float(len(good)/max(len(kp1),1)*100)

def decode_qr(file):
    img = np.array(Image.open(file).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(gray)
    
    return data

def save_history(pid,res,score):
    cursor.execute("INSERT INTO history VALUES(NULL,?,?,?,?)",
                   (pid,res,score,str(datetime.now())))
    conn.commit()

# ---------------- UI STYLE ----------------
st.set_page_config(layout="wide")

st.markdown("""
<style>
            
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

.header {
    background: linear-gradient(90deg, #1f3c88, #4b6cb7);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    border: 2px solid #e0e7ff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

[data-testid="stMetric"] {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border: 2px solid #c7d2fe;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f3c88, #4b6cb7);
}

.stButton>button {
    background: linear-gradient(90deg, #1f3c88, #4b6cb7);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

.stButton>button:hover {
    transform: scale(1.05);
}
/* FIX TEXT VISIBILITY */
label {
    color: black !important;
    font-weight: 600;
}

input, textarea {
    color: black !important;
}

/* Sidebar labels white */
section[data-testid="stSidebar"] label {
    color: white !important;
}            
/* -------- TEXT FIX -------- */
/* -------- COMPLETE TEXT FIX -------- */

.stApp, .stApp * {
    color: #111 !important;
}

input, textarea {
    color: #111 !important;
}

div[role="radiogroup"] label {
    color: #111 !important;
}

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: white !important;
}
/* FIX SELECTED VALUE TEXT */
div[data-baseweb="select"] span {
    color: black !important;
}

/* FIX SELECTBOX INPUT AREA */
div[data-baseweb="select"] > div {
    color: black !important;
    background-color: white !important;
}

/* FIX HIGHLIGHTED OPTION */
div[role="option"] {
    color: black !important;
}

div[role="option"]:hover {
    background-color: #e6f0ff !important;
    color: black !important;
}      
/* SHOW SELECTED VALUE CLEARLY */
div[data-baseweb="select"] span {
    color: black !important;
}

/* INPUT BOX TEXT FIX */
div[data-baseweb="select"] input {
    color: black !important;
}

/* DROPDOWN BOX BACKGROUND */
div[data-baseweb="select"] {
    background-color: white !important;
}                  
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login=False

if not st.session_state.login:
    c1,c2,c3=st.columns([1,2,1])
    with c2:
        st.markdown("<div class='card'><h3>🔐 Login</h3></div>",unsafe_allow_html=True)
        u=st.text_input("Username")
        p=st.text_input("Password",type="password")

        if st.button("Login"):
            if u=="admin" and p=="admin123":
                st.session_state.login=True
            else:
                st.error("Invalid login")
    st.stop()




# ---------------- MENU ----------------
menu = st.sidebar.radio("Menu",
["Dashboard","Register","Verify","View","Edit","Delete","History"])
if menu == "Dashboard":

    st.markdown("<h1 style='text-align:center;'>🛡 Fake Product Detection</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        try:
            st.image("banner.jpg", width=600)
        except:
            st.info("Add banner.jpg")

    cursor.execute("SELECT COUNT(*) FROM products")
    total_products = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history")
    total_scans = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE result='Genuine'")
    genuine = cursor.fetchone()[0]

    fake = total_scans - genuine

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Products", total_products)
    c2.metric("Total Scans", total_scans)
    c3.metric("Genuine", genuine)
    c4.metric("Fake Products", fake)
st.divider()

st.subheader("📊 Analytics")
cursor.execute("SELECT result FROM history")
data = cursor.fetchall()
if data:
    df = pd.DataFrame(data, columns=["Result"])

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df["Result"].value_counts())

    with col2:
        st.line_chart(df["Result"].value_counts())
else:
    st.info("No data for analytics yet")

# ---------------- REGISTER ----------------
if menu=="Register":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📦 Register Product")

    name = st.text_input("Product Name")
    mfg = st.text_input("Manufacturer")
    batch = st.text_input("Batch Number")
    date = st.date_input("Manufacturing Date")

    file = st.file_uploader("Upload Product Image")

    if st.button("Save Product"):
        if not file:
            st.warning("⚠️ Please upload product image")
        elif not name or not mfg or not batch:
            st.warning("⚠️ Fill all details")
        else:
            pid = generate_product_id()
            img = file.getvalue()
            h = generate_hash(img)

            cursor.execute(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
                (pid, name, mfg, batch, str(date), img, h, "0")
            )
            conn.commit()

            qr_data = f"{pid}|{h}"
            qr = generate_qr(qr_data)

            st.success(f"✅ Product Registered Successfully (ID: {pid})")

            col1, col2 = st.columns(2)
            with col1:
                st.image(Image.open(io.BytesIO(img)), caption="Product Image")

            with col2:
                st.image(qr, caption="Generated QR Code")
                st.download_button("⬇ Download QR", qr, f"{pid}.png")

    st.markdown("</div>", unsafe_allow_html=True)
        

# ---------------- VERIFY ----------------
elif menu=="Verify":
    st.markdown("<div class='card'>",unsafe_allow_html=True)

    method=st.radio("Method",["Upload","Camera","QR Upload","QR Camera"])
    pid=st.text_input("Product ID")

    cursor.execute("SELECT * FROM products WHERE id=?", (pid,))
    product=cursor.fetchone()

    if method in ["Upload","Camera"]:
        file=st.file_uploader("Upload") if method=="Upload" else st.camera_input("Camera")

        if st.button("Verify"):
            if not file or not product:
                st.error("Missing data")
            else:
                orb=orb_similarity(product[5],file)
                dl=deep_similarity(product[5],file)
                final=(0.3*orb)+(0.7*dl)

                st.progress(int(final))

                if final>65:
                    st.success("✅ Genuine")
                    save_history(pid,"Genuine",final)
                else:
                    st.error("❌ Fake")
                    save_history(pid,"Fake",final)

    else:
        file=st.file_uploader("QR") if method=="QR Upload" else st.camera_input("Scan QR")

        if st.button("Verify QR") and file and product:
            data=decode_qr(file)
            if data==f"{pid}|{product[6]}":
                st.success("✅ Genuine Product")
            else:
                st.error("❌ Fake Product")

    st.markdown("</div>",unsafe_allow_html=True)

