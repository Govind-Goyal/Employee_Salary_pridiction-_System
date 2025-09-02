import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="Employee Salary Classification",
    page_icon="🧑🏼‍💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------ LOAD MODEL ------------------ #
model = joblib.load("best_model.pkl")

# ------------------ INIT SESSION STATE ------------------ #
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "users" not in st.session_state:
    st.session_state.users = {}  # username: password

# ------------------ AUTH SYSTEM ------------------ #
def sign_up(username, password):
    if username in st.session_state.users:
        return False
    st.session_state.users[username] = password
    return True

def sign_in(username, password):
    return st.session_state.users.get(username) == password

if not st.session_state.authenticated:
    st.title("🔐 Welcome to Salary Classification App")

    auth_tab = st.tabs(["🔓 Sign In", "📝 Sign Up"])

    # ----- SIGN IN TAB -----
    with auth_tab[0]:
        st.subheader("Sign In")
        with st.form("sign_in_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                if sign_in(username, password):
                    st.session_state.authenticated = True
                    st.success("✅ Logged in successfully!")
                    st.experimental_rerun()
                else:
                    st.error("❌ Invalid username or password.")

    # ----- SIGN UP TAB -----
    with auth_tab[1]:
        st.subheader("Sign Up")
        with st.form("sign_up_form"):
            new_user = st.text_input("Choose a Username")
            new_pass = st.text_input("Choose a Password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                if sign_up(new_user, new_pass):
                    st.success("🎉 Account created! You can now sign in.")
                else:
                    st.error("⚠️ Username already exists. Try another one.")
    st.stop()

# ------------------ MAIN APP AFTER LOGIN ------------------ #
st.title("🧑🏼‍💼 Employee Salary Classification")
st.markdown("Use this app to **predict whether an employee earns more than 50K or not** based on their job details.")

tabs = st.tabs(["📋 Single Prediction", "📂 Batch Prediction"])

# ----- SINGLE PREDICTION -----
with tabs[0]:
    st.subheader("Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("👤 Age", 18, 65, 30)
        education = st.selectbox("🎓 Education Level", [
            "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
        ])
        experience = st.slider("🏢 Years of Experience", 0, 40, 5)

    with col2:
        occupation = st.selectbox("💼 Job Role", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
            "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
            "Protective-serv", "Armed-Forces"
        ])
        hours_per_week = st.slider("⏱️ Hours per Week", 1, 80, 40)

    input_df = pd.DataFrame({
        'age': [age],
        'education': [education],
        'occupation': [occupation],
        'hours-per-week': [hours_per_week],
        'experience': [experience]
    })

    st.markdown("#### 🔎 Input Summary")
    st.dataframe(input_df, use_container_width=True)

    if st.button("📊 Predict Salary Class"):
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted Salary Class: `{prediction[0]}`")

# ----- BATCH PREDICTION -----
with tabs[1]:
    st.subheader("📂 Upload CSV for Bulk Predictions")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.markdown("✅ File uploaded. Preview below:")
        st.dataframe(batch_data.head(), use_container_width=True)

        try:
            batch_preds = model.predict(batch_data)
            batch_data['PredictedClass'] = batch_preds
            st.markdown("### 🔮 Predictions")
            st.dataframe(batch_data.head(), use_container_width=True)

            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name="predicted_classes.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")


