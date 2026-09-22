"""Streamlit app: Student Performance Predictor."""

import streamlit as st

from src.predict import load_model, predict_student

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("Student Performance Predictor")
st.write(
    "Predict a student's final math performance from study and lifestyle factors. "
    "Fill in the details below and click **Predict Performance**."
)

# Load the trained pipeline once per session.
model = load_model()

with st.form("student_form"):
    st.subheader("Student Information")

    col1, col2 = st.columns(2)

    with col1:
        school = st.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home Address", ["U", "R"], help="U = urban, R = rural")
        famsize = st.selectbox("Family Size", ["LE3", "GT3"], help="LE3 = 3 or fewer, GT3 = more than 3")
        Pstatus = st.selectbox("Parents Living Together", ["T", "A"], help="T = together, A = apart")
        Medu = st.slider("Mother's Education (0-4)", 0, 4, 2)
        Fedu = st.slider("Father's Education (0-4)", 0, 4, 2)
        Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
        reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])

    with col2:
        traveltime = st.slider("Travel Time to School (1-4)", 1, 4, 1, help="1: <15min, 2: 15-30min, 3: 30min-1h, 4: >1h")
        studytime = st.slider("Weekly Study Time (1-4)", 1, 4, 2, help="1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h")
        failures = st.slider("Past Class Failures", 0, 3, 0)
        schoolsup = st.selectbox("Extra School Support", ["no", "yes"])
        famsup = st.selectbox("Family Educational Support", ["no", "yes"])
        paid = st.selectbox("Extra Paid Classes", ["no", "yes"])
        activities = st.selectbox("Extracurricular Activities", ["no", "yes"])
        nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access at Home", ["yes", "no"])
        romantic = st.selectbox("In a Romantic Relationship", ["no", "yes"])

    st.subheader("Lifestyle")
    col3, col4 = st.columns(2)
    with col3:
        famrel = st.slider("Family Relationship Quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free Time After School (1-5)", 1, 5, 3)
        goout = st.slider("Going Out with Friends (1-5)", 1, 5, 3)
    with col4:
        Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, 1)
        health = st.slider("Current Health (1-5)", 1, 5, 5)
        absences = st.number_input("Absences", 0, 93, 4)

    submitted = st.form_submit_button("Predict Performance")

if submitted:
    inputs = {
        "school": school, "sex": sex, "age": age, "address": address,
        "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
        "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
        "traveltime": traveltime, "studytime": studytime, "failures": failures,
        "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences,
    }
    result = predict_student(model, inputs)

    st.subheader("Prediction")
    st.markdown(f"### **{result['prediction']} Performance**")

    st.subheader("Confidence")
    for class_name, probability in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        st.write(f"{class_name}: {probability:.1%}")

    st.info(
        "This prediction is an educational machine learning demonstration and should not be used as a definitive assessment of a student's academic ability."
    )
