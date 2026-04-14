import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="نظام التعرف على الصقور", page_icon="🦅", layout="centered")

# 2. تنسيق اللغة العربية والألوان (CSS)
st.markdown("""
    <style>
    .main {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stMarkdownContainer"] > p {
        text-align: right;
        direction: rtl;
    }
    /* تنسيق أداة رفع الملفات */
    div[data-testid="stFileUploader"] {
        direction: rtl;
        text-align: right;
        border: 2px dashed #1e3a8a;
        border-radius: 15px;
        padding: 10px;
    }
    /* تنسيق صندوق الإقرار */
    .disclaimer {
        background-color: #fff3cd;
        border-right: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin-top: 30px;
        color: #856404;
        font-size: 0.9em;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. عرض الشعار (تأكدي من رفع ملف logo.jpg في GitHub)
try:
    st.image("logo.jpg", width=120)
except:
    pass # في حال لم يتم رفع الشعار بعد، لن يظهر خطأ

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

# 7. أداة رفع الصور (بدون نص مكرر)
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

# 8. الإقرار العلمي (في أسفل الصفحة)
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ تنبيه علمي:</strong>  

    هذا النموذج (Model) لا يزال تحت التحسين والتطوير، وقد يظهر تحيزاً لبعض أنواع الصقور بناءً على بيانات التدريب المتاحة. 
    لذا، يفضل دائماً الاستعانة بصقارين خبراء للتأكد من النتائج، وذلك لتعزيز وتأكيد دقة التصميم في المراحل القادمة.
</div>
""", unsafe_allow_html=True)
