import streamlit as st

# ----- NHẬP THƯ VIỆN VÀ FILE HỖ TRỢ CẦN THIẾT -----
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px

# For EDA and Text Preprocessing
from wordcloud import WordCloud, STOPWORDS
import string
import os
import re
from deep_translator import GoogleTranslator
from langdetect import detect
from underthesea import word_tokenize, pos_tag, sent_tokenize
import emoji

# For Sentiment Analyst
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from nltk.probability import FreqDist
from sklearn.utils import resample

# For Text Clustering
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

import joblib
from joblib import dump
from joblib import load

import gdown

import warnings
warnings.filterwarnings('ignore')

# ----- TIỀN XỬ LÝ VĂN BẢN CHO SENTIMENT ANALYSIS -----
def preprocess_review_text(df, col_like, col_suggestion):
    # Load emojicon
    with open('emojicon.txt', 'r', encoding="utf8") as file:
        emoji_dict = {line.split('\t')[0]: line.split('\t')[1] for line in file.read().split('\n') if '\t' in line}

    # Load teencode
    with open('teencode.txt', 'r', encoding="utf8") as file:
        teen_dict = {line.split('\t')[0]: line.split('\t')[1] for line in file.read().split('\n') if '\t' in line}

    # Load English to Vietnamese dictionary
    with open('english-vnmese.txt', 'r', encoding="utf8") as file:
        english_dict = {line.split('\t')[0]: line.split('\t')[1] for line in file.read().split('\n') if '\t' in line}

    # Load wrong word list
    with open('wrong-word-2.txt', 'r', encoding="utf8") as file:
        wrong_lst = file.read().split('\n')

    # Load stopwords
    with open('vietnamese-stopwords.txt', 'r', encoding="utf8") as file:
        stopwords_lst = file.read().split('\n')

    def smart_translate_langdetect(text):
        try:
            lang = detect(text)
            if lang == 'en':
                return GoogleTranslator(source='en', target='vi').translate(text)
            else:
                return text
        except:
            return text

    def process_special_word(text):
        special_words = ['không', 'chẳng', 'chả', 'chưa', 'thiếu', 'hơi']
        new_text = ''
        text_lst = text.split()
        i = 0
        while i < len(text_lst):
            word = text_lst[i]
            if word in special_words and i + 1 < len(text_lst):
                combined = word + '_' + text_lst[i + 1]
                new_text += combined + ' '
                i += 2
            else:
                new_text += word + ' '
                i += 1
        return new_text.strip()

    def loaddicchar():
        uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
        unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
        dic = {}
        char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
        charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
        for i in range(len(char1252)):
            dic[char1252[i]] = charutf8[i]
        return dic

    def covert_unicode(txt):
        dicchar = loaddicchar()
        return re.sub('|'.join(dicchar.keys()), lambda x: dicchar[x.group()], txt)

    def clean_text_SA(text):
        text = text.lower()
        text = text.replace("'", '')
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'([a-z]+?)\1+', r'\1', text)
        new_sentence = ''
        for sentence in sent_tokenize(text):
            sentence = ''.join(emoji_dict.get(word, word) + ' ' if word in emoji_dict else word for word in list(sentence))
            sentence = ' '.join(teen_dict.get(word, word) for word in sentence.split())
            pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
            sentence = ' '.join(re.findall(pattern, sentence))
            sentence = sentence.replace('.', '')
            lst_word_type = ['A','AB','V','VB','VY','R']
            sentence = ' '.join(word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
            new_sentence += sentence + '. '
        text = new_sentence
        english_dict['environment'] = 'moi_truong'
        english_dict['ot'] = 'tang_ca'
        text = ' '.join(english_dict.get(word, word) for word in text.split())
        stopwords_lst.append('cong_ty')
        text = ' '.join(word for word in text.split() if word not in stopwords_lst)
        text = ' '.join(word for word in text.split() if word not in wrong_lst)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Process translation
    df[col_like + '_translated'] = df[col_like].apply(smart_translate_langdetect)
    df[col_suggestion + '_translated'] = df[col_suggestion].apply(smart_translate_langdetect)

    # Fill NA
    df[col_like + '_translated'].fillna('na', inplace=True)
    df[col_suggestion + '_translated'].fillna('na', inplace=True)

    # Unicode normalization
    df[col_like + '_translated'] = df[col_like + '_translated'].apply(covert_unicode)
    df[col_suggestion + '_translated'] = df[col_suggestion + '_translated'].apply(covert_unicode)

    # Clean text
    df['like_cleaned'] = df[col_like + '_translated'].apply(clean_text_SA)
    df['suggestion_cleaned'] = df[col_suggestion + '_translated'].apply(clean_text_SA)

    # Combine both
    df['review_cleaned'] = df['like_cleaned'] + ' ' + df['suggestion_cleaned']

    return df[['review_cleaned']]

# ----- Load model của Sentiment Analysis -----
# best_rf_pipeline = joblib.load('rf_tfidf_pipeline_sentiment.joblib')
url = "https://drive.google.com/uc?id=1HT6uH8Q-RylS-Fl-9SmrOowRc4xg4ReG"
output = "rf_tfidf_pipeline_sentiment_2.joblib"  # Đổi tên theo nhu cầu
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Kiểm tra dung lượng file
if os.path.exists(output):
    size_kb = os.path.getsize(output) / 1024
    # st.write(f"✅ File tải về: {output} ({size_kb:.2f} KB)")
    if size_kb < 100:  # Bạn có thể điều chỉnh ngưỡng này
        st.error("⚠️ File mô hình tải về có vẻ không đúng hoặc bị lỗi.")
    else:
        best_rf_pipeline = joblib.load(output)
else:
    st.error("❌ File mô hình không tồn tại sau khi tải.")

# ---- Load data cho text clustering ------
data = pd.read_excel('Reviews.xlsx')
data_TC = pd.read_csv('Reviews_cleaned_for_TC_v2.csv')
data_topic = data_TC.copy()

# ----- Load model của Text Clustering -----
pipeline_like = joblib.load('pipeline_like.pkl')
# Trích xuất từng bước
vectorizer_like = pipeline_like.named_steps['vectorizer']
lda_like = pipeline_like.named_steps['lda']
kmeans_like = pipeline_like.named_steps['kmeans']
# Transform từng bước
like_vectorizer = vectorizer_like.transform(data_topic['like_cleaned'])
like_topic_dist = lda_like.transform(like_vectorizer)
like_cluster = kmeans_like.predict(like_topic_dist)

pipeline_suggestion = joblib.load('pipeline_suggestion.pkl')
# Trích xuất từng bước
vectorizer_suggestion = pipeline_suggestion.named_steps['vectorizer']
lda_suggestion = pipeline_suggestion.named_steps['lda']
kmeans_suggestion = pipeline_suggestion.named_steps['kmeans']
# Transform từng bước
suggestion_vectorizer = vectorizer_suggestion.transform(data_topic['suggestion_cleaned'])
suggestion_topic_dist = lda_suggestion.transform(suggestion_vectorizer)
suggestion_cluster = kmeans_suggestion.predict(suggestion_topic_dist)

data_topic['like_topic'] = like_cluster
data_topic['suggestion_topic'] = suggestion_cluster

# ----- Recommendation Mapping for clustering -----
RECOMMEND_TOPIC_MAP = {
    0: "🏢 Cải thiện không gian làm việc và cơ sở vật chất.\nGợi ý: nâng cấp văn phòng, chỗ ngồi, khu vực nghỉ ngơi, thiết bị.",

    1: "⚙️ Nâng cấp quy trình & chính sách nội bộ và hoạt động nhóm.\nGợi ý: đơn giản hóa thủ tục nội bộ, cải tiến hệ thống quản lý, tăng minh bạch.",

    2: "💸 Cải thiện chế độ tăng ca, lương, thưởng và đãi ngộ.\nGợi ý: xem xét chế độ tăng ca, điều chỉnh lương, thưởng theo hiệu suất, tăng hỗ trợ tài chính."
}

LIKE_TOPIC_MAP = {
    0: "🏢 Không gian làm việc & Cơ sở vật chất",
    1: "📈 Cơ hội phát triển & Văn hóa công ty",
    2: "💰 Phúc lợi & Đãi ngộ & Đồng nghiệp",
}

# Tạo DataFrame chỉ chứa ID công ty và suggestion_topic
company_suggestions = data_topic[['id', 'like_topic', 'suggestion_topic']]

# Sau đó, lấy ra topic có số lần xuất hiện nhiều nhất
def get_top_n_suggestions(group, n=2):
  if len(group) == 0:
    return []
  topic_counts = group['suggestion_topic'].value_counts()
  # Lấy tối đa n topic có tần suất cao nhất
  top_topics_suggestion = topic_counts.head(n).index.tolist()
  return top_topics_suggestion

def get_top_n_like(group, n=1):
  if len(group) == 0:
    return []
  topic_counts = group['like_topic'].value_counts()
  # Lấy tối đa n topic có tần suất cao nhất
  top_topics_like = topic_counts.head(n).index.tolist()
  return top_topics_like

company_top_suggestions = company_suggestions.groupby('id').apply(get_top_n_suggestions).reset_index(name='top_suggestions')
company_top_likes = company_suggestions.groupby('id').apply(get_top_n_like).reset_index(name='top_likes')

# Chuyển danh sách các topic suggestions sang tên gợi ý dựa vào mapping
company_top_likes['top_like_names'] = company_top_likes['top_likes'].apply(
    lambda topics: [LIKE_TOPIC_MAP.get(topic, "Không rõ chủ đề") for topic in topics]
)
company_top_suggestions['top_suggestion_names'] = company_top_suggestions['top_suggestions'].apply(
    lambda topics: [RECOMMEND_TOPIC_MAP.get(topic, "Không rõ chủ đề") for topic in topics]
)

### ----- Back-up ------
# Hàm lấy top n topic cho 1 cột
def get_top_n(group, column, n=1):
    if group.empty:
        return []
    return group[column].value_counts().head(n).index.tolist()

# Nhóm theo ID rồi lấy top topic cho cả 2 cột trong cùng 1 apply
company_top_topics = company_suggestions.groupby('id').apply(
    lambda g: pd.Series({
        'top_like': get_top_n(g, 'like_topic'),
        'top_suggestion': get_top_n(g, 'suggestion_topic')
    })
).reset_index()

# Chuyển danh sách các topic like và suggestions sang tên gợi ý dựa vào mapping
company_top_topics['top_like_names'] = company_top_topics['top_like'].apply(
    lambda topics: [LIKE_TOPIC_MAP.get(topic, "Không rõ chủ đề") for topic in topics]
)

company_top_topics['top_suggestion_names'] = company_top_topics['top_suggestion'].apply(
    lambda topics: [RECOMMEND_TOPIC_MAP.get(topic, "Không rõ chủ đề") for topic in topics]
)

# ----- Streamlit App -----

# Thiết lập tiêu đề chính
st.set_page_config(page_title="Ứng dụng Demo", layout="wide")

# Thanh menu bên trái (sidebar)
with st.sidebar:
    st.title("Menu")

    # Chọn trang mục
    page = st.radio("Chọn trang", ["1. Giới thiệu", "2. Phân tích & Kết quả", "3. Phân tích cảm xúc", "4. Phân nhóm đánh giá"])

    # Dòng phân cách
    st.markdown("---")

    # Thông tin nhóm
    st.markdown("**Thành viên nhóm:**")
    st.markdown("- Mr. Lê Đức Anh")
    st.markdown("- Mr. Trần Anh Tú")

    # Giáo viên hướng dẫn
    st.markdown("**GVHD:**")
    st.markdown("- Ms. Khuất Thùy Phương")

    # Khoảng trống đẩy nội dung xuống
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

     # Dòng chữ nhỏ ở dưới cùng
    st.markdown(
        "<div style='font-size: 11px; color: gray; text-align: center;'>"
        "Dự án tốt nghiệp<br>Data Science & Machine Learning<br>TTTH - ĐH KHTN"
        "</div>",
        unsafe_allow_html=True)

# Hiển thị nội dung theo từng trang
if page == "1. Giới thiệu":
    # Hiển thị banner ITviec
    st.image("banner_itviec_3.jpg", caption='Nguồn: ITviec', use_container_width=True)
    st.header("1. Giới thiệu")

    # Giới thiệu ITviec
    st.subheader("Về ITviec")
    st.markdown("""
    ITViec là nền tảng chuyên cung cấp các cơ hội việc làm trong lĩnh vực Công nghệ Thông tin (IT) hàng đầu Việt Nam.
    Nền tảng này được thiết kế để giúp người dùng, đặc biệt là các developer, phát triển sự nghiệp một cách hiệu quả.
    Người dùng có thể dễ dàng tìm kiếm việc làm trên ITViec theo nhiều tiêu chí khác nhau như kỹ năng, chức danh và công ty.
    Bên cạnh đó, ITViec còn cung cấp nhiều tài nguyên hữu ích hỗ trợ người tìm việc và phát triển bản thân, bao gồm:
    """)
    st.markdown("""
    - **Đánh giá công ty**: Giúp ứng viên có cái nhìn tổng quan về môi trường làm việc và văn hóa của các công ty IT.
    - **Blog chuyên ngành**: Chia sẻ các bài viết về kiến thức chuyên môn, kỹ năng mềm, xu hướng công nghệ và các lời khuyên nghề nghiệp hữu ích.
    - **Báo cáo lương IT**: Cung cấp thông tin về mức lương trên thị trường, giúp người dùng có cơ sở để đàm phán mức đãi ngộ phù hợp.
    """)

    # Giới thiệu Dataset
    st.subheader("Về bộ dữ liệu")
    st.markdown("""
    Bộ dữ liệu bao gồm **hơn 8.000 đánh giá** từ các nhân viên và cựu nhân viên trong ngành IT tại Việt Nam, được thu thập từ ITviec.com.

    Các trường chính:
    - `What I liked`: Những điều tích cực người đánh giá cảm nhận.
    - `Suggestions for improvement`: Gợi ý cải thiện dành cho công ty.
    - `Company Mame`, `id`, `Recommend?`, `Overall Rating`, v.v...

    Dữ liệu đã được xử lý tiếng Việt, chuẩn hóa và làm sạch trước khi áp dụng mô hình học máy.
    """)

    # Mục tiêu của ứng dụng
    st.subheader("Mục tiêu của ứng dụng")
    st.markdown("""
    Ứng dụng này được xây dựng với hai mục tiêu chính:

    1. **Phân tích cảm xúc (Sentiment Analysis)**: Dự đoán cảm xúc của người đánh giá (Good / Neutral / Bad) dựa trên nội dung họ cung cấp.

    2. **Phân nhóm nội dung đánh giá (Information Clustering)**: Tự động phân loại và trực quan hóa các đánh giá theo chủ đề, giúp doanh nghiệp hiểu rõ các điểm mạnh và điểm cần cải thiện.
    """)

elif page == "2. Phân tích & Kết quả":
    st.header("2. Phân tích và Kết quả Mô hình")
    # 1. Mô tả dữ liệu
    st.subheader("2.1 Khám phá dữ liệu ban đầu")
    st.markdown("- Số lượng dòng dữ liệu: 8417")
    st.markdown("- Số lượng cột: 12")
    st.markdown("- Trường thông tin: `What I liked`, `Suggestions for improvement`, `Rating`")
    # Hiển thị vài dòng đầu tiên
    st.dataframe(data.head())

    col1, col2 = st.columns(2)
    with col1:
      st.markdown("#### Làm sạch")
      st.markdown("""
      - **Drop duplicates**: 5 entries  
      - **Drop null values**  
      - **Drop rows where `What I liked` is null**  
      - **Drop rows where `Suggestion for improvement` is null**  
      """)

    with col2:
      st.markdown("#### Tổng quan dữ liệu")
      st.markdown("""
        - **`What I liked` feature**  
        - Mean: 237 digits  
        - Max: 6384 digits  
        - **`Suggestion for improvement` feature**  
        - Mean: 138 digits  
        - Max: 3813 digits  
        """)
    st.image("EDA_length.png", caption="Độ dài kí tự")

    # 2. Tiền xử lý văn bản
    st.subheader("2.2 Tiền xử lý văn bản")
    
    st.markdown("""
    #### 1. Dịch ngôn ngữ
    - Phát hiện ngôn ngữ với `langdetect`
    - Dịch tiếng Anh sang tiếng Việt bằng `GoogleTranslator`

    #### 2. Mã hóa & Chuẩn hóa ký tự
    - Mã hóa chuẩn `UTF-8`
    - Chuyển toàn bộ văn bản thành chữ **thường**
    - Loại bỏ ký tự **lặp lại** (vd: *thiệtttt* → *thiệt*)

    #### 3. Xử lý Emoji & Teen Code
    - **Sentiment:** Thay emoji bằng từ mô tả (`emojicon.txt`)
    - **Clustering:** Phát hiện và xóa emoji
    - Chuyển teen code sang từ thông dụng (`teencode.txt`)

    #### 4. Làm sạch văn bản
    - Loại bỏ **dấu câu** và **chữ số**
    - Chuẩn hóa **Unicode** từ các dạng gõ lỗi

    #### 5. POS Tag & Từ đặc biệt
    - Gắn nhãn từ loại bằng `underthesea.pos_tag`
    - Ghép từ đặc biệt: *không, chưa, chả...* → *không_tốt*, *chưa_ổn*

    #### 6. Lọc từ theo mục tiêu
    - **Sentiment:** giữ từ loại `['A','AB','V','VB','VY','R']`
    - **Clustering:** giữ thêm danh từ `['N', 'Np', 'A', 'AB', 'V', 'VB', 'VY', 'R]`

    #### 7. Dịch từ đơn
    - Dùng `english-vnmese.txt` để dịch các từ đơn còn sót

    #### 8. Loại bỏ nhiễu
    - Xoá stopwords (`vietnamese-stopwords.txt`)
    - Xoá từ sai (`wrong-word-2.txt`)
    - Xoá khoảng trắng thừa
    """)

    st.markdown("### Word Cloud của tệp data cho Sentiment Analysis")
    col3, col4 = st.columns(2)
    with col3:
        st.image("word_cloud_like_sentiment.png", caption="WordCloud - What I liked") #use_container_width=True
    with col4:
        st.image("word_cloud_suggestion_sentiment.png", caption="WordCloud - Suggestions for improvement") #use_container_width=True


    st.markdown("### Word Cloud của tệp data cho Information Clustering")
    col5, col6 = st.columns(2)
    with col5:
        st.image("word_cloud_like_cluster.png", caption="WordCloud - What I liked") #use_container_width=True
    with col6:
        st.image("word_cloud_suggestion_cluster.png", caption="WordCloud - Suggestions for improvement") #use_container_width=True

    # 3. Trích xuất đặc trưng và mô hình hóa
    st.subheader("2.3 Mô hình phân tích cảm xúc")
    st.markdown("- Tiến hành **Resample** để cân bằng dữ liệu biến `label`")
    st.markdown("- Sử dụng mô hình **Random Forest** với đầu vào là TF-IDF vector")
    st.markdown("- Độ chính xác: **97.77%**")
    # Hiển thị hình ảnh kết quả các model
    st.image("sentiment_analysis_results.png", caption="Model Comparison")
    # Hiển thị biểu đồ confusion matrix nếu có
    st.image("confusion_matrix_sentiment_RF.png", caption="Confusion Matrix Random Forest")

    # 4. Phân tích chủ đề bằng LDA và K Means
    st.subheader("2.4 Phân tích chủ đề")
    st.markdown("- Sử dụng **LDA** và 3 mô hình `KMeans`,  `Agglomerative Clustering`, `Gaussian Mixture`, và so sánh kết quả")
    st.markdown("- Kết quả tốt nhất là sử dụng **LDA + KMeans** để phân nhóm theo chủ đề review")
    st.markdown("- Số lượng chủ đề (LDA) và cụm (KMeans) đều là 3 cho mỗi phần `What I liked` và `Suggestions for improvement`")

    # Hiển thị hình ảnh biểu đồ tam giác
    st.markdown("- Silhoutte Score - What I liked: **0.59**")
    st.image("ternary_like.png", caption="Biểu đồ phân bổ - LIKE")
    st.markdown("- Silhoutte Score - What I liked: **0.64**")
    st.image("ternary_suggestion.png", caption="Biểu đồ phẩn bổ - SUGGESTION")

    # Kết quả cluster - LIKE
    st.markdown("### **Kết quả cluster 'What I liked'**")
    col7, col8, col9 = st.columns(3)
    # Hiển thị ảnh trong từng cột
    with col7:
        st.image("topic_0_like.png", caption="Top 10 keywords - chủ đề 0") # use_container_width=True

    with col8:
        st.image("topic_1_like.png", caption="Top 10 keywords - chủ đề 1") # use_container_width=True

    with col9:
        st.image("topic_2_like.png", caption="Top 10 keywords - chủ đề 2") # use_container_width=True

    # Cluster 0
    st.markdown("#### 🏢 Cluster 0: Không gian làm việc & Cơ sở vật chất (chủ đề 1)")
    st.markdown("""
    **Từ khóa tiêu biểu:** `văn_phòng`, `đẹp`, `đội`, `phòng`, `máy`  
    """)

    # Cluster 1
    st.markdown("#### 📈 Cluster 1: Cơ hội phát triển & Văn hóa công ty (chủ đề 2)")
    st.markdown("""
    **Từ khóa tiêu biểu:** `phát_triển`, `chính_sách`, `văn_hóa`  
    """)

    # Cluster 2
    st.markdown("#### 💰 Cluster 2: Phúc lợi, Đãi ngộ & Đồng nghiệp (chủ đề 0)")
    st.markdown("""
    **Từ khóa tiêu biểu:** `lương`, `tăng_ca`, `chế_độ`, `dự_án`, `sếp`, `đồng_nghiệp`
    """)

    # Kết quả cluster - SUGGESTION
    st.markdown("### **Kết quả cluster 'Suggestions for improvement'**")
    col10, col11, col12 = st.columns(3)
    # Hiển thị ảnh trong từng cột
    with col10:
        st.image("topic_0_suggestion.png", caption="Top 10 keywords - chủ đề 0") # use_container_width=True

    with col11:
        st.image("topic_1_suggestion.png", caption="Top 10 keywords - chủ đề 1") # use_container_width=True

    with col12:
        st.image("topic_2_suggestion.png", caption="Top 10 keywords - chủ đề 2") # use_container_width=True

    # Cluster 0
    st.markdown("#### 🏢 Cluster 0: Cải thiện không gian & cở sở vật chất (chủ đề 0)")
    st.markdown("""
    **Từ khóa tiêu biểu:** `văn_phòng`, `phòng`, `chỗ`, `đội`, `trưa`, `hoạt_động`   
    """)

    # Cluster 1
    st.markdown("#### ⚙️ Cluster 1: Nâng cấp quy trình & chính sách nội bộ và hoạt động nhóm (chủ đề 1)")
    st.markdown("""
    **Từ khóa tiêu biểu:** `cải_thiện`, `đội`, `chính_sách`, `phát_triển`, `dự án`, `cải_thiện` 
    """)

    # Cluster 2
    st.markdown("#### 💸 Cluster 2: Tăng ca, lương, thưởng & đãi ngộ (chủ đề 2)")
    st.markdown("""
    **Từ khóa tiêu biểu:** `tăng_ca`, `lương`, `tiền`, `thưởng`, `sếp`, `chậm`
    """)

# Sentiment Analysis
elif page == "3. Phân tích cảm xúc":
    st.header("3. Phân tích cảm xúc")
    st.write("Nhập tay hoặc tải file lên để dự đoán")

    # --- Nhóm 1: Nhập tay ---
    st.subheader("Nhập tay đánh giá")
    liked_input = st.text_area("What I liked")
    suggestion_input = st.text_area("Suggestions for improvement")

    if st.button("Dự đoán",  key="predict_manual"):
        if liked_input.strip() == "" and suggestion_input.strip() == "":
            st.warning("Vui lòng nhập ít nhất một trường.")
        else:
            try:
                df_input = pd.DataFrame({
                    "What I liked": [liked_input],
                    "Suggestions for improvement": [suggestion_input]
                })
                df_input = preprocess_review_text(df_input, "What I liked", "Suggestions for improvement")
                preds = best_rf_pipeline.predict(df_input['review_cleaned'])

                label_mapping = {0: 'Bad', 1: 'Neutral', 2: 'Good'}
                pred_label = label_mapping.get(preds[0], "Unknown")

                st.success(f"Sentiment dự đoán: **{pred_label}**")
            except Exception as e:
                st.error(f"Lỗi xảy ra: {e}")

    st.markdown("---")

    # --- Nhóm 2: Upload file ---
    st.subheader("Upload file Excel (.xlsx) hoặc CSV (.csv)")
    uploaded_file = st.file_uploader("Chọn file Excel hoặc CSV chứa 2 cột: 'What I liked' và 'Suggestions for improvement'", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df_upload = pd.read_excel(uploaded_file)
            else:
                df_upload = pd.read_csv(uploaded_file)

            # Kiểm tra cột cần thiết
            required_cols = ['What I liked', 'Suggestions for improvement']
            if all(col in df_upload.columns for col in required_cols):
                if st.button("Dự đoán", key="predict_from_file"):
                    df_cleaned = preprocess_review_text(df_upload, "What I liked", "Suggestions for improvement")
                    preds = best_rf_pipeline.predict(df_cleaned['review_cleaned'])

                    label_mapping = {0: 'Bad', 1: 'Neutral', 2: 'Good'}
                    df_upload['Predicted Sentiment'] = [label_mapping.get(p, "Unknown") for p in preds]

                    st.success("Dự đoán hoàn tất! Kết quả hiển thị bên dưới:")
                    st.dataframe(df_upload[['What I liked', 'Suggestions for improvement', 'Predicted Sentiment']])

                    # Tạo và tải file kết quả
                    output_csv = df_upload[['What I liked', 'Suggestions for improvement', 'Predicted Sentiment']].to_csv(index=False).encode('utf-8-sig')
                    st.download_button("Tải xuống file kết quả", data=output_csv, file_name='sentiment_analysis_results.csv', mime='text/csv')
            else:
                st.error(f"File thiếu cột: {required_cols}")
        except Exception as e:
            st.error(f"Lỗi khi xử lý file: {e}")


elif page == "4. Phân nhóm đánh giá":
    st.header("4. Phân nhóm đánh giá")
    # st.write("Lựa chọn công ty cần tìm hiểu")

    # Tạo danh sách dropdown "ID - Tên công ty"
    company_options = data_topic[['id', 'Company Name']].drop_duplicates()
    company_options['display_name'] = company_options['id'].astype(str) + " - " + company_options['Company Name']

    selected_display = st.selectbox("Chọn công ty", company_options['display_name'].tolist())

    if selected_display:
        # Trích xuất ID công ty từ chuỗi chọn
        company_id_input = int(selected_display.split(" - ")[0])

        try:
            # --- LIKE TRIANGLE ---
            df_like = pd.DataFrame(like_topic_dist, columns=["Topic 0", "Topic 1", "Topic 2"])
            df_like = pd.concat([df_like, data_topic['id']], axis=1)
            df_like = df_like[df_like['id'] == company_id_input]
            df_like["Cluster"] = data_topic['like_topic'].astype(str)

            st.subheader("Đánh giá phân loại công ty")
            fig_like = px.scatter_ternary(
                df_like,
                a="Topic 0", b="Topic 1", c="Topic 2",
                color="Cluster",
                size_max=10,
                opacity=0.8,
                title="Biểu đồ phân bổ đánh giá 'What I liked'",
                labels={
                    "Topic 0": "Phúc lợi & Đãi ngộ & Đồng nghiệp",
                    "Topic 1": "Không gian làm việc & Cơ sở vật chất",
                    "Topic 2": "Cơ hội phát triển & Văn hóa công ty",
                    "Cluster": "Cụm"
                }
            )
            st.plotly_chart(fig_like)

            # --- SUGGESTION TRIANGLE ---
            df_suggestion = pd.DataFrame(suggestion_topic_dist, columns=["Topic 0", "Topic 1", "Topic 2"])
            df_suggestion = pd.concat([df_suggestion, data_topic['id']], axis=1)
            df_suggestion = df_suggestion[df_suggestion['id'] == company_id_input]
            df_suggestion["Cluster"] = data_topic['suggestion_topic'].astype(str)

            # st.subheader("Đánh giá phân loại công ty")
            fig_sugg = px.scatter_ternary(
                df_suggestion,
                a="Topic 0", b="Topic 1", c="Topic 2",
                color="Cluster",
                size_max=10,
                opacity=0.8,
                title="Biểu đồ phân bổ đánh giá 'Suggestions for improvement'",
                labels={
                    "Topic 0": "Cải thiện không gian & cơ sở vật chất",
                    "Topic 1": "Quy trình, chính sách & teamwork",
                    "Topic 2": "Tăng ca, lương, thưởng & đãi ngộ",
                    "Cluster": "Cụm"
                }
            )
            st.plotly_chart(fig_sugg)

            # --- GỢI Ý CẢI THIỆN ---
            st.subheader("Top ưu điểm của công ty")
            row_like = company_top_likes[company_top_likes['id'] == company_id_input]
            row_suggestion = company_top_suggestions[company_top_suggestions['id'] == company_id_input]

            if not row_like.empty:
                for like in row_like['top_like_names'].iloc[0]:
                    st.markdown(f"**{like}**")
            else:
                st.warning("Không tìm thấy thông tin ưu điểm cho công ty này.")

            st.subheader("Top đề xuất cải thiện")
            if not row_suggestion.empty:
                for suggestion in row_suggestion['top_suggestion_names'].iloc[0]:
                    st.markdown(f"**{suggestion}**")
            else:
                st.warning("Không tìm thấy thông tin đề xuất cho công ty này.")

        except Exception as e:
            st.error(f"Có lỗi xảy ra: {e}")

