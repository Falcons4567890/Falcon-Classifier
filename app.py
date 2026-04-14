import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="نظام التعرف على الصقور", page_icon="🦅", layout="centered")

# 2. تنسيق الهوية (إطار بني وبيج، زخارف تراثية، وتنسيق عربي كامل)
st.markdown("""
    <style>
    /* إطار الصفحة الكاملة مع زخرفة تراثية */
    .stApp {
        border: 15px solid #5d4037;
        background-image: 
            radial-gradient(circle at 0 0, transparent 0, transparent 20px, #5d4037 21px, #5d4037 25px, transparent 26px),
            radial-gradient(circle at 100% 0, transparent 0, transparent 20px, #5d4037 21px, #5d4037 25px, transparent 26px),
            radial-gradient(circle at 100% 100%, transparent 0, transparent 20px, #5d4037 21px, #5d4037 25px, transparent 26px),
            radial-gradient(circle at 0 100%, transparent 0, transparent 20px, #5d4037 21px, #5d4037 25px, transparent 26px);
        background-size: 50px 50px;
        background-repeat: no-repeat;
        background-position: 0 0, 100% 0, 100% 100%, 0 100%;
        min-height: 100vh;
        direction: rtl;
        text-align: right;
        background-color: #fafafa;
    }
    
    /* شريط زخرفي علوي */
    .stApp::before {
        content: "۞ ═══════════════════════ ۞ ═══════════════════════ ۞";
        display: block;
        text-align: center;
        color: #8d6e63;
        font-size: 20px;
        padding: 10px 0;
        letter-spacing: 5px;
    }

    .main {
        padding: 20px 40px;
    }

    /* تنسيق النصوص والعناوين باللون البني */
    h1, h2, h3, p, label {
        text-align: right !important;
        direction: rtl !important;
        color: #4e342e !important;
    }

    /* إطار أداة رفع الصور بلون بيج غامق */
    div[data-testid="stFileUploader"] {
        direction: rtl;
        text-align: right;
        border: 2px dashed #8d6e63 !important;
        border-radius: 15px;
        background-color: #efebe9;
        padding: 20px;
    }
    
    div[data-testid="stFileUploader"] label {
        text-align: right;
        width: 100%;
    }

    /* إعادة صندوق الإقرار العلمي للون الأصفر والذهبي الأصلي */
    .disclaimer {
        background-color: #fff3cd; /* اللون الأصفر الأصلي */
        border-right: 8px solid #ffc107; /* الحافة الذهبية الأصلية */
        padding: 25px;
        border-radius: 10px;
        margin-top: 30px;
        color: #856404; /* لون النص البني المائل للذهبي */
        font-size: 1.1em;
        line-height: 1.8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: right;
        direction: rtl;
        position: relative;
    }
    
    /* زخرفة صغيرة داخل صندوق الإقرار */
    .disclaimer::after {
        content: "❃";
        position: absolute;
        bottom: 5px;
        left: 10px;
        color: #ffc107;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. عرض الشعار في جهة اليمين (تأكدي من رفع logo.jpg)
try:
    col_logo, col_empty1, col_empty2 = st.columns([1, 1, 1])
    with col_logo:
        st.image("logo.jpg", use_container_width=True)
except:
    pass

# 4. العنوان والوصف
st.markdown('<h1 style="text-align: right; direction: rtl;">نظام التعرف الذكي على الصقور 🦅</h1>', unsafe_allow_html=True)
st.write("ارفع صورة الصقر أو قم بلصقها هنا للحصول على التحليل والمعلومات فوراً.")

# 5. تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 6. قاعدة بيانات المعلومات
falcon_info = {
    "Shaheen": {
        "title": "🦅 صقر الشاهين (Peregrine Falcon)",
        "details": """الاسم العلمي: Falco peregrinus  
  
الحالة: مهاجر جزئيًا، مع وجود سلالات مستوطنة  
  
الموائل: السواحل والمرتفعات والبيئات المفتوحة  
  
ميزة مميزة: أسرع كائن حي أثناء الانقضاض"""
    },
    "Gyer": {
        "title": "🦅 صقر الجير (Gyrfalcon)",
        "details": """الاسم العلمي: Falco rusticolus  
  
الحالة: زائر شتوي نادر  
  
الموائل: المناطق القطبية وشبه القطبية، السهول المفتوحة والتندرا  
  
ميزة مميزة: أكبر أنواع الصقور وأقواها، يتميز بقوة الانقضاض وقدرته على الصيد في البيئات القاسية"""
    },
    "Hur": {
        "title": "🦅 الصقر الحر (Saker Falcon)",
        "details": """الاسم العلمي: Falco cherrug  
  
الحالة: مهاجر شتوي  
  
الموائل: السهول المفتوحة والبيئات الصحراوية  
  
ميزة مميزة: رمز الصقارة التقليدية وقوة التحمل"""
    },
    "Wakri": {
        "title": "🦅 الصقر الوكري (Lanner Falcon)",
        "details": """الاسم العلمي: Falco biarmicus  
  
الحالة: مستوطن  
  
الموائل: البيئات شبه الجافة والصخرية  
  
ميزة مميزة: صياد مرن بأساليب مطاردة متنوعة"""
    }
}

# 7. أداة رفع الصور
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="الصورة الأصلية", use_container_width=True)
    
    with st.spinner('جاري تحليل الصورة...'):
        results = model.predict(source=image, conf=0.25, imgsz=640)
        res_plotted = results[0].plot(labels=False)
        result_img = Image.fromarray(res_plotted[:, :, ::-1])
        
        with col2:
            st.image(result_img, caption="النتيجة", use_container_width=True)
        
        if len(results[0].boxes) > 0:
            class_id = int(results[0].boxes.cls[0])
            eng_name = model.names[class_id].split('_')[0]
            
            if eng_name in falcon_info:
                info = falcon_info[eng_name]
                st.markdown(f"""
                <div style="text-align: right; direction: rtl; padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-right: 5px solid #3182ce;">
                    <h3 style="margin-top: 0;">{info['title']}</h3>
                    <p style="font-size: 1.1em; line-height: 1.6;">{info['details']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("لم يتم التعرف على نوع الصقر بدقة، حاول مرة أخرى بصورة أوضح.")

# 8. الإقرار العلمي
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ تنبيه علمي:</strong>  

    هذا النموذج الذكي لا يزال تحت التحسين والتطوير، وقد يظهر تحيزاً لبعض أنواع الصقور بناءً على بيانات التدريب المتاحة. 
    لذا، يفضل دائماً الاستعانة بصقارين خبراء للتأكد من النتائج، وذلك لتعزيز وتأكيد دقة التصميم في المراحل القادمة.
</div>
""", unsafe_allow_html=True)
