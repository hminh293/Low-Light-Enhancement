import streamlit as st 
from models.gan import Generator
from models.mirnetv2_model import MIRNetV2
from models.Retinex import RetinexNet
import torch
from PIL import Image
import numpy as np
import tempfile
import cv2
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]  # project root (one level up)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Kiểm tra thiết bị một lần duy nhất
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(model_name):
    def load_state(model, path):
        # Đưa model lên device ngay khi load
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

    if model_name == "GAN":
        return load_state(Generator(), 'demo/best_model_state/GAN.pth')
    elif model_name == "MIR":
        return load_state(MIRNetV2(), 'demo/best_model_state/MIR.pth')
    elif model_name == "RET":
        return load_state(RetinexNet(), 'demo/best_model_state/RET.pth')

def transform_image(input_data, mode="image"):
    if mode == "image":
        # input_data là đối tượng file
        low_img = Image.open(input_data).convert('RGB')
        low_img = np.array(low_img)
    else:
        # input_data là numpy array từ OpenCV (BGR)
        low_img = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    
    # Resize sau khi đã chuyển sang RGB
    low_img = cv2.resize(low_img, (256, 256))
    low_img = low_img.astype(np.float32) / 255.0
    low_img = low_img.transpose(2, 0, 1) # HWC to CHW
    return torch.from_numpy(low_img)

def enhance_image(model, low_img_tensor):
    # Đảm bảo tensor nằm trên cùng device với model
    low_img_tensor = low_img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced_img_tensor = model(low_img_tensor)
    
    # Quan trọng: .cpu() để có thể convert sang numpy
    enhanced_img_tensor = enhanced_img_tensor.squeeze(0).cpu().clamp(0, 1)
    enhanced_img = enhanced_img_tensor.permute(1, 2, 0).numpy()
    return (enhanced_img * 255).astype(np.uint8)

# --- GIAO DIỆN ---
st.set_page_config(page_title="Low-Light Enhancement", layout="wide")
st.title("🌙 Low-Light Image Enhancement Demo")

model_name = st.selectbox("Chọn Model", ["GAN", "MIR", "RET"])
model = load_model(model_name)

st.sidebar.title("Cài đặt")
app_mode = st.sidebar.selectbox("Chế độ đầu vào", ["IMAGE", "VIDEO", "WEBCAM"])

if app_mode == "IMAGE":
    uploaded_file = st.file_uploader("Tải ảnh...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        low_img_tensor = transform_image(uploaded_file, mode="image")
        enhanced_img = enhance_image(model, low_img_tensor)
        
        c1, c2 = st.columns(2)
        c1.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
        c2.image(enhanced_img, caption="Đã xử lý", use_container_width=True)

elif app_mode == "VIDEO":
    video_file = st.file_uploader("Tải video...", type=["mp4", "mov", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        col1 , col2 = st.columns(2)
        with col1:
            st.subheader("Video Gốc")
            original_placeholder = st.empty() # Khung cho video gốc
            
        with col2:
            st.subheader("Video Đã Tăng Sáng")
            enhanced_placeholder = st.empty() # Khung cho video đã xử lý
        
        # Nút dừng video
        stop = st.button("Dừng video")
        
        while vf.isOpened() and not stop:
            ret, frame = vf.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Xử lý frame-by-frame
            low_img_frame = transform_image(frame, mode="video")
            enhanced_frame = enhance_image(model, low_img_frame)
            
            # Hiển thị
            
            original_placeholder.image(frame_rgb, use_container_width=True)
            enhanced_placeholder.image(enhanced_frame, use_container_width=True)

elif app_mode == "WEBCAM":
    st.info("Đang mở luồng Webcam trực tiếp từ máy tính...")
    
    # Tạo 2 cột để hiển thị song song
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Webcam Gốc")
        webcam_original = st.empty()
    with col2:
        st.subheader("Webcam Đã Xử Lý")
        webcam_enhanced = st.empty()

    # Mở webcam (0 là camera mặc định)
    cap = cv2.VideoCapture(0)
    
    # Nút dừng webcam
    stop_webcam = st.button("Dừng Webcam")

    while cap.isOpened() and not stop_webcam:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập Webcam")
            break

        # 1. Chuyển màu frame gốc để hiển thị (OpenCV đọc là BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. Đưa qua Model xử lý (Dùng hàm transform với mode video)
        low_img_tensor = transform_image(frame, mode="video")
        enhanced_frame = enhance_image(model, low_img_tensor)

        # 3. Hiển thị lên 2 cột cùng lúc
        webcam_original.image(frame_rgb, use_container_width=True)
        webcam_enhanced.image(enhanced_frame, use_container_width=True)

    cap.release()