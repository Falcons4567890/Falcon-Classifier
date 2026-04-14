import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="نظام التعرف على الصقور", page_icon="🦅", layout="centered")

# 2. تنسيق الهوية (إطار بني وذهبي وتنسيق عربي)
st.markdown("""
    <style>
    .stApp {
        border: 15px solid #5d4037;
        border-image: linear-gradient(to bottom, #5d4037, #c5a059) 1;
        min-height: 100vh;
        direction: rtl;
        text-align: right;
    }
    .main { padding: 40px; }
    h1, h2, h3, p, label { text-align: right !important; direction: rtl !important; color: #3e2723 !important; }
    
    /* تنسيق أداة الرفع لتكون بنية */
    div[data-testid="stFileUploader"] {
        direction: rtl;
        text-align: right;
        border: 3px solid #5d4037 !important;
        border-radius: 15px;
        background-color: #efebe9;
        padding: 20px;
    }

    /* إجبار صندوق التنبيه على الظهور كاملاً بدون أي Scroll */
    div[data-testid="stNotification"] {
        direction: rtl;
        text-align: right;
        overflow: visible !important;
        height: auto !important;
        min-height: 150px;
    }
    div[data-testid="stNotification"] div {
        overflow: visible !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. عرض الشعار في المنتصف تماماً
try:
    col_empty1, col_logo, col_empty2 = st.columns([1, 1, 1])
    with col_logo:
        st.image("logo.jpg", use_container_width=True)
except:
    pass

# 4. العنوان والوصف
st.markdown('<h1 style="text-align: center; direction: rtl;">نظام التعرف الذكي على الصقور 🦅</h1>', unsafe_allow_html=True)
st.write("<p style='text-align: center;'>ارفع صورة الصقر أو قم بلصقها هنا للحصول على التحليل والمعلومات فوراً.</p>", unsafe_allow_html=True)

# 5. تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# 6. قاعدة بيانات المعلومات
falcon_info = {
    "Shaheen": {"title": "🦅 صقر الشاهين (Peregrine Falcon)", "details": "الاسم العلمي: Falco peregrinus  
  
الحالة: مهاجر جزئيًا، مع وجود سلالات مستوطنة  
  
الموائل: السواحل والمرتفعات والبيئات المفتوحة  
  
ميزة مميزة: أسرع كائن حي أثناء الانقضاض"},
    "Gyer": {"title": "🦅 صقر الجير (Gyrfalcon)", "details": "الاسم العلمي: Falco rusticolus  
  
الحالة: زائر شتوي نادر  
  
الموائل: المناطق القطبية وشبه القطبية، السهول المفتوحة والتندرا  
  
ميزة مميزة: أكبر أنواع الصقور وأقواها"},
    "Hur": {"title": "🦅 الصقر الحر (Saker Falcon)", "details": "الاسم العلمي: Falco cherrug  
  
الحالة: مهاجر شتوي  
  
الموائل: السهول المفتوحة والبيئات الصحراوية  
  
ميزة مميزة: رمز الصقارة التقليدية وقوة التحمل"},
    "Wakri": {"title": "🦅 الصقر الوكري (Lanner Falcon)", "details": "الاسم العلمي: Falco biarmicus  
  
الحالة: مستوطن  
  
الموائل: البيئات شبه الجافة والصخرية  
  
ميزة مميزة: صياد مرن بأساليب مطاردة متنوعة"}
}

# 7. أداة رفع الصور
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1: st.image(image, caption="الصورة الأصلية", use_container_width=True)
    
    with st.spinner('جاري تحليل الصورة...'):
        results = model.predict(source=image, conf=0.25, imgsz=640)
        res_plotted = results[0].plot(labels=False)
        result_img = Image.fromarray(res_plotted[:, :, ::-1])
        with col2: st.image(result_img, caption="النتيجة", use_container_width=True)
        
        if len(results[0].boxes) > 0:
            class_id = int(results[0].boxes.cls[0])
            eng_name = model.names[class_id].split('_')[0]
            if eng_name in falcon_info:
                info = falcon_info[eng_name]
                st.markdown(f"""<div style="text-align: right; direction: rtl; padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-right: 5px solid #3182ce;"><h3>{info['title']}</h3><p>{info['details']}</p></div>""", unsafe_allow_html=True)
        else:
            st.warning("لم يتم التعرف على نوع الصقر بدقة، حاول مرة أخرى بصورة أوضح.")

            # 8. الإقرار العلمي (باستخدام أداة Streamlit الرسمية مع منع الـ Scroll)
st.markdown("---")
st.warning("""
⚠️ تنبيه علمي:
هذا النموذج الذكي لا يزال تحت التحسين والتطوير، وقد يظهر تحيزاً لبعض أنواع الصقور بناءً على بيانات التدريب المتاحة. لذا، يفضل دائماً الاستعانة بصقارين خبراء للتأكد من النتائج، وذلك لتعزيز وتأكيد دقة التصميم في المراحل القادمة.
""")
