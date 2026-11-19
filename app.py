
import os
import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix



project_dir = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(project_dir, "tuned_random_forest_model.pkl")
)

scaler = joblib.load(
    os.path.join(project_dir, "scaler.pkl")
)

encoder = joblib.load(
    os.path.join(project_dir, "encoder.pkl")
)

metadata = joblib.load(
    os.path.join(project_dir, "project_metadata.pkl")
)

numerical_columns = metadata["numerical_columns"]
categorical_columns = metadata["categorical_columns"]
columns = metadata["columns"]
categorical_options = metadata["categorical_options"]
numerical_defaults = metadata["numerical_defaults"]



st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Scoring System")

st.markdown(
    "### Machine Learning Based Credit Risk Prediction"
)

st.write(
    "Enter the applicant's financial and personal details "
    "to generate a credit risk prediction."
)

st.info(
    "Educational machine learning demonstration only. "
    "Not intended for actual lending decisions."
)



display_names = {
    "Attribute2": "Credit Duration (Months)",
    "Attribute5": "Credit Amount",
    "Attribute8": "Installment Rate (%)",
    "Attribute11": "Years at Current Residence",
    "Attribute13": "Applicant Age",
    "Attribute16": "Number of Existing Credits",
    "Attribute18": "People Liable for Maintenance"
}


with st.form("applicant_form"):

    applicant_data = {}

    st.subheader("Applicant Financial Information")

    for col in numerical_columns:

        label = display_names.get(col, col)

        applicant_data[col] = st.number_input(
            label,
            value=float(numerical_defaults[col]),
            min_value=0.0,
            step=1.0
        )

    st.divider()

    st.subheader("Additional Applicant Information")

    for col in categorical_columns:

        options = categorical_options[col]

        applicant_data[col] = st.selectbox(
            col,
            options=options
        )

    submitted = st.form_submit_button(
        "Predict Credit Risk",
        use_container_width=True
    )



if submitted:

    applicant_df = pd.DataFrame(
        [applicant_data],
        columns=columns
    )

    applicant_num = scaler.transform(
        applicant_df[numerical_columns]
    )

    applicant_cat = encoder.transform(
        applicant_df[categorical_columns]
    )

    applicant_processed = hstack([
        csr_matrix(applicant_num),
        applicant_cat
    ])

    prediction = model.predict(
        applicant_processed
    )[0]

    probabilities = model.predict_proba(
        applicant_processed
    )[0]

    bad_index = list(model.classes_).index(2)
    good_index = list(model.classes_).index(1)

    bad_probability = probabilities[bad_index] * 100
    good_probability = probabilities[good_index] * 100

    

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Category: Good Credit")
    else:
        st.warning("Predicted Category: Bad Credit")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Bad Credit Probability",
            f"{bad_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Good Credit Probability",
            f"{good_probability:.2f}%"
        )

    st.progress(
        int(round(bad_probability))
    )

    st.caption(
        "These are model-estimated probabilities, "
        "not necessarily calibrated real-world probabilities."
    )

    with st.expander("View entered applicant details"):
        st.dataframe(
            applicant_df,
            use_container_width=True
        )
