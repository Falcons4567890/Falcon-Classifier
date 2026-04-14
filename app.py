import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# إعدادات الصفحة (العنوان والأيقونة)
st.set_page_config(page_title="نظام التعرف على الصقور", page_icon="🦅", layout="centered")

# تنسيق اللغة العربية (من اليمين لليسار)
st.markdown("""
    <style>
    .reportview-container { direction: rtl; }
    .main { text-align: right; }
    div.stButton > button:first-child { background-color: #3182ce; color: white; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# تحميل الموديل (مع التخزين المؤقت للسرعة)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# قاعدة بيانات المعلومات
falcon_info = {
    "Shaheen": {
        "title": "🦅 صقر الشاهين (Peregrine Falcon)",
        "details": "الاسم العلمي: Falco peregrinus  
الحالة: مهاجر جزئيًا، مع وجود سلالات مستوطنة  
الموائل: السواحل والمرتفعات والبيئات المفتوحة  
ميزة مميزة: أسرع كائن حي أثناء الانقضاض"
    },
    "Gyer": {
        "title": "🦅 صقر الجير (Gyrfalcon)",
        "details": "الاسم العلمي: Falco rusticolus  
الحالة: زائر شتوي نادر  
الموائل: المناطق القطبية وشبه القطبية، السهول المفتوحة والتندرا  
ميزة مميزة: أكبر أنواع الصقور وأقواها"
    },
    "Hur": {
        "title": "🦅 الصقر الحر (Saker Falcon)",
        "details": "الاسم العلمي: Falco cherrug  
الحالة: مهاجر شتوي  
الموائل: السهول المفتوحة والبيئات الصحراوية  
ميزة مميزة: رمز الصقارة التقليدية وقوة التحمل"
    },
    "Wakri": {
        "title": "🦅 صقر الوكري (Lanner Falcon)",
        "details": "الاسم العلمي: Falco biarmicus  
الحالة: مستوطن  
الموائل: البيئات شبه الجافة والصخرية  
ميزة مميزة: صياد مرن بأساليب مطاردة متنوعة"
    }
}

st.title("🦅 نظام التعرف الذكي على الصقور")
st.write("ارفع صورة الصقر للحصول على النتيجة والمعلومات فوراً.")

uploaded_file = st.file_uploader("اختر صورة صقر...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    
    if st.button('تحليل الصورة الآن'):
        with st.spinner('جاري التحليل...'):
            # تشغيل التوقع
            results = model.predict(source=image, conf=0.25, imgsz=320)
            res_plotted = results[0].plot(labels=False)
            
            st.image(res_plotted[:, :, ::-1], caption='النتيجة', use_column_width=True)
            
            if len(results[0].boxes) > 0:
                class_id = int(results[0].boxes.cls[0])
                eng_name = model.names[class_id].split('_')[0]
                
                if eng_name in falcon_info:
                    info = falcon_info[eng_name]
                    st.markdown(f"""
                    <div style="text-align: right; direction: rtl; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
                        <h3>{info['title']}</h3>
                        <p>{info['details']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("لم يتم التعرف على نوع الصقر بدقة.")