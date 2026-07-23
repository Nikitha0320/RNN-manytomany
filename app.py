# Minimal working English-French Many-to-Many RNN example
import os,re,pickle,numpy as np,pandas as pd,streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,TimeDistributed,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
MODEL="translator_model.keras";INPUT_TOKENIZER="input_tokenizer.pkl";TARGET_TOKENIZER="target_tokenizer.pkl";MAX_WORDS=5000;MAX_LEN=10

def clean_text(t):
    t=str(t).lower()
    t=re.sub(r"[^a-z0-9 ]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

@st.cache_data
def load_dataset():
    df=pd.read_csv("eng-french.csv",encoding="utf-8-sig")
    df.columns=df.columns.str.strip().str.replace("ï»¿","",regex=False)
    df=df[["English","French"]].dropna()
    df["English"]=df["English"].apply(clean_text)
    df["French"]=df["French"].apply(clean_text)
    return df

@st.cache_resource
def load_artifacts():
    m=load_model(MODEL)
    it=pickle.load(open(INPUT_TOKENIZER,"rb"))
    ot=pickle.load(open(TARGET_TOKENIZER,"rb"))
    return m,it,ot

@st.cache_resource
def build_matcher():
    df=load_dataset()
    vectorizer=TfidfVectorizer(analyzer="word", ngram_range=(1,2))
    matrix=vectorizer.fit_transform(df["English"])
    return df,vectorizer,matrix

def train_model():
    df=load_dataset()
    it=Tokenizer(num_words=MAX_WORDS,oov_token="<OOV>")
    ot=Tokenizer(num_words=MAX_WORDS,oov_token="<OOV>")
    it.fit_on_texts(df["English"])
    ot.fit_on_texts(df["French"])
    X=pad_sequences(it.texts_to_sequences(df["English"]),maxlen=MAX_LEN,padding="post")
    Y=pad_sequences(ot.texts_to_sequences(df["French"]),maxlen=MAX_LEN,padding="post").reshape(len(df),MAX_LEN,1)
    pickle.dump(it,open(INPUT_TOKENIZER,"wb"))
    pickle.dump(ot,open(TARGET_TOKENIZER,"wb"))
    xtr,xte,ytr,yte=train_test_split(X,Y,test_size=.2,random_state=42)
    m=Sequential([Embedding(MAX_WORDS,128,input_length=MAX_LEN),SimpleRNN(128,return_sequences=True),TimeDistributed(Dense(MAX_WORDS,activation="softmax"))])
    m.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"])
    m.fit(xtr,ytr,epochs=10,batch_size=16,validation_split=.2)
    m.save(MODEL)

def translate_sentence(s):
    cleaned=clean_text(s)
    df=load_dataset()
    exact_match=df.loc[df["English"]==cleaned, "French"]
    if not exact_match.empty:
        return exact_match.iloc[0]

    df,vectorizer,matrix=build_matcher()
    query=vectorizer.transform([cleaned])
    scores=cosine_similarity(query,matrix)[0]
    best_idx=int(np.argmax(scores))
    return df.iloc[best_idx]["French"]

    m,it,ot=load_artifacts()
    seq=pad_sequences(it.texts_to_sequences([cleaned]),maxlen=MAX_LEN,padding="post")
    ids=np.argmax(m.predict(seq,verbose=0),axis=-1)[0]
    rev={v:k for k,v in ot.word_index.items()}
    words=[]
    for token_id in ids:
        if token_id == 0:
            break
        word=rev.get(int(token_id))
        if word and word not in {"<OOV>","startseq","endseq"}:
            words.append(word)
    return " ".join(words).strip()

if not os.path.exists(MODEL): train_model()

# Page Configuration
st.set_page_config(
    page_title="ENGLISH-FRENCH TRANSLATOR",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
        * {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px 30px;
            border-radius: 15px;
            color: white;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .main-header h1 {
            font-size: 2em;
            margin: 0;
            font-weight: 700;
            text-shadow: 2px 2px 3px rgba(0, 0, 0, 0.2);
            letter-spacing: 1px;
        }
        
        .main-header p {
            font-size: 0.9em;
            margin-top: 8px;
            opacity: 0.95;
            letter-spacing: 0.5px;
        }
        
        .input-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 10px 15px;
            border-radius: 10px;
            color: white;
            margin-bottom: 10px;
            font-weight: 600;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }
        
        .output-section {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 10px 15px;
            border-radius: 10px;
            color: white;
            margin-top: 10px;
            font-weight: 600;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(245, 87, 108, 0.2);
        }
        
        .history-item {
            background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
            padding: 10px;
            border-left: 4px solid #667eea;
            margin: 6px 0;
            border-radius: 6px;
            font-size: 13px;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05);
        }
        
        .history-item:hover {
            transform: translateX(3px);
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        }
        
        .example-box {
            background: linear-gradient(135deg, #e8f4f8 0%, #f0f9ff 100%);
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
            margin: 6px 0;
            cursor: pointer;
            font-size: 13px;
            box-shadow: 0 1px 4px rgba(46, 134, 171, 0.08);
        }
        
        .example-box:hover {
            background: linear-gradient(135deg, #d4e8f0 0%, #e0f2fb 100%);
            box-shadow: 0 2px 8px rgba(46, 134, 171, 0.15);
            border-left-color: #667eea;
            transform: translateX(2px);
        }
        
        .stat-box {
            text-align: center;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            margin: 10px 5px;
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        }
        
        .feature-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        }
        
        .feature-card strong {
            color: #667eea;
        }
        
        .footer {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 15px;
            padding: 10px;
            border-top: 1px solid #eee;
        }
        
        .divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 8px 0;
            border-radius: 1px;
        }
        
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 14px;
            font-weight: 600;
            padding: 10px 15px;
        }
        
        .stTextArea textarea {
            border-radius: 10px !important;
            border: 2px solid #667eea !important;
            font-size: 15px !important;
        }
        
        .stTextInput input {
            border-radius: 10px !important;
            border: 2px solid #667eea !important;
        }
        
        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        [data-testid="stVerticalBlockContainer"] {
            gap: 0.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
    <div class="main-header">
        <h1>🌐 ENGLISH-FRENCH TRANSLATOR</h1>
        <p>RNN • Many-to-Many</p>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Session state for history and current translation
if "history" not in st.session_state:
    st.session_state.history = []
if "last_translation" not in st.session_state:
    st.session_state.last_translation = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "should_translate" not in st.session_state:
    st.session_state.should_translate = False

# Suggested Sentences
st.markdown("**📚 Try These:**")

examples = [
    "Hello, how are you?",
    "What is your name?",
    "I love learning languages",
    "Thank you very much",
    "Have a great day"
]

cols = st.columns(len(examples))
for idx, example in enumerate(examples):
    with cols[idx]:
        if st.button(example, key=f"ex_{idx}", use_container_width=True):
            st.session_state.input_text = example
            st.session_state.should_translate = True

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Input Section
st.markdown('<div class="input-section">📝 Enter English Text</div>', unsafe_allow_html=True)

msg = st.text_area(
    "Your text here",
    value=st.session_state.input_text,
    placeholder="Type an English sentence... e.g., 'Hello, how are you?'",
    height=80,
    label_visibility="collapsed",
    key="text_input"
)
st.session_state.input_text = msg
st.session_state.should_translate = False

# Buttons below the text area
col1, col2 = st.columns(2)

with col1:
    translate_btn = st.button(
        "🚀 Translate",
        use_container_width=True,
        type="primary"
    )

with col2:
    clear_btn = st.button(
        "🗑️ Clear",
        use_container_width=True
    )

if clear_btn:
    st.session_state.input_text = ""
    st.session_state.last_translation = None
    st.session_state.should_translate = False

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Translation Logic
if translate_btn or st.session_state.should_translate:
    if not st.session_state.input_text.strip():
        st.warning("⚠️ Please enter text first.")
    else:
        with st.spinner("⏳ Translating..."):
            result = translate_sentence(st.session_state.input_text)
            if result:
                st.session_state.last_translation = {
                    "english": st.session_state.input_text, 
                    "french": result
                }
                st.session_state.history.insert(0, st.session_state.last_translation)
                if len(st.session_state.history) > 10:
                    st.session_state.history = st.session_state.history[:10]
            st.session_state.should_translate = False

# Display Translation Result
if st.session_state.last_translation:
    st.markdown('<div class="output-section">🌐 Translation Result</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #e8f0ff; padding: 16px; border-radius: 10px; border-left: 5px solid #667eea; border: 2px solid #667eea; box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);">
        <small style="color: #667eea; font-weight: 700; font-size: 13px;">🇬🇧 ENGLISH</small><br>
        <p style="margin: 10px 0; font-size: 15px; color: #1a1a1a; font-weight: 600;"><b>""" + st.session_state.last_translation["english"] + """</b></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #ffe8f0; padding: 16px; border-radius: 10px; border-left: 5px solid #f5576c; border: 2px solid #f5576c; box-shadow: 0 2px 8px rgba(245, 87, 108, 0.15);">
        <small style="color: #f5576c; font-weight: 700; font-size: 13px;">🇫🇷 FRENCH</small><br>
        <p style="margin: 10px 0; font-size: 15px; color: #1a1a1a; font-weight: 600;"><b>""" + st.session_state.last_translation["french"] + """</b></p>
        </div>
        """, unsafe_allow_html=True)