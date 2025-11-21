import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================
# LOAD & TRAIN MODELS (CACHE)
# ============================================
@st.cache_resource
def load_and_train():
    # Load dataset
    df = pd.read_csv("BankNote_Authentication.csv")

    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Model A - Logistic Regression
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Model B - Random Forest
    rf = Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ))
    ])

    logreg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    acc_logreg = accuracy_score(y_test, logreg.predict(X_test))
    acc_rf = accuracy_score(y_test, rf.predict(X_test))

    return df, logreg, rf, acc_logreg, acc_rf


# ============================================
# START PAGE
# ============================================
st.set_page_config(page_title="Banknote ML App", page_icon="💵")

st.title("💵 Banknote Authentication ML App")
st.write(
    """
    Aplikasi ini memprediksi **uang asli atau palsu** menggunakan 2 model:
    - Logistic Regression  
    - Random Forest  
    """
)

# Load models
df, logreg_model, rf_model, acc_logreg, acc_rf = load_and_train()


# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("🔧 Pengaturan")

chosen_model = st.sidebar.selectbox(
    "Pilih Model:",
    ("Logistic Regression", "Random Forest")
)

st.sidebar.subheader("📊 Akurasi Model")
st.sidebar.metric("Logistic Regression", f"{acc_logreg*100:.2f}%")
st.sidebar.metric("Random Forest", f"{acc_rf*100:.2f}%")


# ============================================
# INPUT FORM
# ============================================
st.subheader("📝 Input Data Baru")

variance = st.number_input("Variance", -10.0, 10.0, 0.0)
skewness = st.number_input("Skewness", -20.0, 20.0, 0.0)
curtosis = st.number_input("Curtosis", -20.0, 20.0, 0.0)
entropy = st.number_input("Entropy", -10.0, 10.0, 0.0)

if st.button("🔍 Prediksi"):
    X_new = pd.DataFrame([{
        "variance": variance,
        "skewness": skewness,
        "curtosis": curtosis,
        "entropy": entropy
    }])

    model = logreg_model if chosen_model == "Logistic Regression" else rf_model

    pred = model.predict(X_new)[0]

    if pred == 0:
        st.success("✅ Hasil: **Uang ASLI**")
    else:
        st.error("⚠️ Hasil: **Uang PALSU**")

    st.write("### Input yang Dimasukkan")
    st.table(X_new)


st.markdown("---")
st.caption("Dibuat dengan Streamlit — Machine Learning App")
