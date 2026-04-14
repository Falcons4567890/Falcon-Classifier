import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="نظام التعرف على الصقور", page_icon="🦅", layout="centered")

# 2. تنسيق اللغة العربية والألوان (CSS) المحدث
st.markdown("""
    <style>
    /* إطار الصفحة الكاملة بألوان البني والبيج */
    .stApp {
        border: 12px solid #5d4037; /* بني غامق فخم */
        border-image: linear-gradient(to bottom, #5d4037, #d7ccc8) 1; /* تدرج من البني للبيج */
        min-height: 100vh;
        direction: rtl;
        text-align: right;
        background-color: #fafafa; /* خلفية بيضاء مائلة للبيج الفاتح جداً */
    }
    
    .main {
        padding: 40px;
    }

    /* تنسيق النصوص والعناوين باللون البني */
    h1, h2, h3, p, label {
        text-align: right !important;
        direction: rtl !important;
        color: #4e342e !important; /* بني داكن للنصوص */
    }

    /* إطار أداة رفع الصور بلون بيج غامق */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #8d6e63 !important;
        border-radius: 15px;
        background-color: #efebe9; /* خلفية بيج فاتحة للأداة */
    }

    /* تنسيق صندوق الإقرار العلمي بألوان متناسقة */
    .disclaimer {
        background-color: #fbe9e7; /* بيج مائل للدفء */
        border-right: 8px solid #8d6e63; /* حافة بنية */
        padding: 25px;
        border-radius: 10px;
        margin-top: 30px;
        color: #4e342e;
        font-size: 1.1em;
        line-height: 1.8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. عرض الشعار في جهة اليمين (تعديل الترتيب)
try:
    # نضع الشعار في العمود الأول (الذي سيمثل اليمين مع التنسيق العربي)
    col_empty1, col_logo, col_empty2 = st.columns([1, 1, 1])
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
        "details": """الاسم العلمي: Falco peregrinus<br><br>الحالة: مهاجر جزئيًا، مع وجود سلالات مستوطنة<br><br>الموائل: السواحل والمرتفعات والبيئات المفتوحة<br><br>ميزة مميزة: أسرع كائن حي أثناء الانقضاض"""
    },
    "Gyer": {
        "title": "🦅 صقر الجير (Gyrfalcon)",
        "details": """الاسم العلمي: Falco rusticolus<br><br>الحالة: زائر شتوي نادر<br><br>الموائل: المناطق القطبية وشبه القطبية، السهول المفتوحة والتندرا<br><br>ميزة مميزة: أكبر أنواع الصقور وأقواها، يتميز بقوة الانقضاض وقدرته على الصيد في البيئات القاسية"""
    },
    "Hur": {
        "title": "🦅 الصقر الحر (Saker Falcon)",
        "details": """الاسم العلمي: Falco cherrug<br><br>الحالة: مهاجر شتوي<br><br>الموائل: السهول المفتوحة والبيئات الصحراوية<br><br>ميزة مميزة: رمز الصقارة التقليدية وقوة التحمل"""
    },
    "Wakri": {
        "title": "🦅 الصقر الوكري (Lanner Falcon)",
        "details": """الاسم العلمي: Falco biarmicus<br><br>الحالة: مستوطن<br><br>الموائل: البيئات شبه الجافة والصخرية<br><br>ميزة مميزة: صياد مرن بأساليب مطاردة متنوعة"""
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

# 8. الإقرار العلمي (نسخة عربية مرتبة بالكامل)
st.markdown("""
<div class="disclaimer">
    <strong>⚠️ تنبيه علمي:</strong><br>
    هذا النموذج الذكي لا يزال تحت التحسين والتطوير، وقد يظهر تحيزاً لبعض أنواع الصقور بناءً على بيانات التدريب المتاحة. 
    لذا، يفضل دائماً الاستعانة بصقارين خبراء للتأكد من النتائج، وذلك لتعزيز وتأكيد دقة التصميم في المراحل القادمة.
</div>
""", unsafe_allow_html=True)
