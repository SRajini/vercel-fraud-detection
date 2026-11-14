"""Streamlit demo.  Run:  streamlit run app.py   (after training the model)"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Claim Fraud Risk", page_icon="🔍")
st.title("🔍 Insurance Claim Fraud Risk Scorer")

art = joblib.load("outputs/fraud_model.joblib")
model, defaults, options = art["model"], art["defaults"], art["options"]

st.caption("Fill in key claim details. Other fields use typical (median/most common) values.")
c1, c2 = st.columns(2)
row = dict(defaults)
row["total_claim_amount"] = c1.number_input("Total claim amount", 0, 200000, 50000)
row["policy_annual_premium"] = c2.number_input("Annual premium", 100, 5000, 1200)
row["days_bind_to_incident"] = c1.number_input("Days from policy start to incident", 0, 10000, 1000)
row["months_as_customer"] = c2.number_input("Months as customer", 0, 600, 200)
row["witnesses"] = c1.number_input("Witnesses", 0, 10, 1)
row["number_of_vehicles_involved"] = c2.number_input("Vehicles involved", 1, 10, 1)
for col, label in [("incident_severity", "Incident severity"),
                   ("incident_type", "Incident type"),
                   ("police_report_available", "Police report available"),
                   ("property_damage", "Property damage")]:
    if col in options:
        row[col] = st.selectbox(label, options[col], index=options[col].index(defaults[col]))
row["claim_to_premium"] = row["total_claim_amount"] / max(row["policy_annual_premium"], 1)

if st.button("Score claim"):
    X = pd.DataFrame([row])[art["features"]]
    p = model.predict_proba(X)[0, 1]
    st.metric("Fraud probability", f"{p:.1%}")
    if p >= art["threshold"]:
        st.error("High risk: send to investigator for review.")
    else:
        st.success("Lower risk: standard processing.")
    st.caption(f"Model: {art['name']} | decision threshold: {art['threshold']:.2f}")
