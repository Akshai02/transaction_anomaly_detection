import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import pymysql
from datetime import datetime


conn=pymysql.connect(host="localhost",user="root",password="1234",db="transaction_anomaly_detection")
cursor=conn.cursor()

def main():
    st.title("Transaction Anomaly Detection")
    st.subheader("Transaction Monitoring System")
    image=Image.open("Anomaly-Detection-img.jpg")
    st.image(image,width=800)
    st.write("Enter transaction details to identify unusual transaction behavior.")
    st.info("An anomaly indicates unusual behavior identified by the model "
            "and does not necessarily represent confirmed fraud.")
    transaction_id = st.text_input("Transaction ID")

    transaction_amount = st.number_input("Transaction Amount",min_value=0.0)
    transaction_volume = st.number_input("Transaction Volume",min_value=0)
    average_transaction_amount = st.number_input("Average Transaction Amount",
                                                 min_value=0.0)
    frequency_of_transactions = st.number_input("Frequency of Transactions",
                                                min_value=0)
    time_since_last_transaction = st.number_input("Time Since Last Transaction",
                                                min_value=0.0)
    income = st.number_input("Income",min_value=0.0)
    transaction_time = st.time_input("Time_of_Day")
    account_type = st.selectbox("Account Type",["Current","Savings"])
    day_of_week = st.selectbox("Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

    #Feature Engineering
    if average_transaction_amount == 0:
        amount_deviation_ratio = 0
    else:
        amount_deviation_ratio = (
                transaction_amount /
                average_transaction_amount
        )

    if income == 0:
        transaction_income_ratio = 0
    else:
        transaction_income_ratio = (
                transaction_amount /
                income
        )

    amount_deviation = (
            transaction_amount -
            average_transaction_amount
    )

    #Time Feature Engineering
    hour=transaction_time.hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    numerical_features=[transaction_amount,transaction_volume,average_transaction_amount,
                        frequency_of_transactions,time_since_last_transaction,
                        income,amount_deviation,amount_deviation_ratio,transaction_income_ratio
                        ,hour_sin,hour_cos]

    numerical_features=pd.DataFrame([numerical_features])
    #categorical features
    categorical_features=pd.DataFrame({"Day_of_Week":[day_of_week],
                                       "Account_Type":[account_type]})


    model1=pickle.load(open("model_IF.sav","rb"))
    model2=pickle.load(open("model_LOF_Deploy.sav","rb"))
    scaler=pickle.load(open("scaler_model.sav","rb"))
    encode=pickle.load(open("encode_model.sav","rb"))
    pred=st.button("Analyse Transaction")
    if pred:

        categorical_encoded=pd.get_dummies(categorical_features,drop_first=True)
        categorical_encoded=categorical_encoded.reindex(columns=encode,fill_value=0)
        scaled_numerical=scaler.transform(numerical_features)
        final_features=np.hstack([scaled_numerical,categorical_encoded])
        Anomaly_Label=model1.predict(final_features)
        Anomaly_Label=int(Anomaly_Label[0])
        LOF_Label=model2.predict(final_features)
        LOF_Label=int(LOF_Label[0])
        Anomaly_Score=float(model1.decision_function(final_features)[0])
        LOF_Score=float(model2.decision_function(final_features)[0])
        if Anomaly_Label== -1 and LOF_Label == -1:

            Risk_Level = "High Risk"
            Consistency_Result = "Consistent - Both Models Anomalous"

        elif Anomaly_Label== -1 and LOF_Label == 1:

            Risk_Level = "Medium Risk"
            Consistency_Result = "Inconsistent - IF Only"

        elif Anomaly_Label == 1 and LOF_Label == -1:

            Risk_Level = "Medium Risk"
            Consistency_Result = "Inconsistent - LOF Only"

        else:

            Risk_Level = "Low Risk"
            Consistency_Result = "Consistent - Both Models Normal"
        st.subheader("Model Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Isolation Forest")
            st.write(f"Label: {Anomaly_Label}")
            st.write(f"Score: {Anomaly_Score:.6f}")

            if Anomaly_Label == -1:
                st.error("Isolation Forest: Anomaly")
            else:
                st.success("Isolation Forest: Normal")

        with col2:
            st.write("### Local Outlier Factor")
            st.write(f"Label: {LOF_Label}")
            st.write(f"Score: {LOF_Score:.6f}")

            if LOF_Label == -1:
                st.error("LOF: Anomaly")
            else:
                st.success("LOF: Normal")
        st.subheader("Consistency Check")

        st.write(Consistency_Result)

        if Risk_Level == "High Risk":
            st.error("Both models detected this transaction as anomalous.")
        elif Risk_Level == "Medium Risk":
            st.warning("The two models disagree. Further investigation is recommended.")
        else:
            st.success("Both models consider this transaction normal.")

        query = """
                INSERT INTO prediction_history
                (
                    Transaction_ID,
                    Transaction_Amount,
                    Anomaly_Score,
                    LOF_Score,
                    Anomaly_Label,
                    LOF_Label,
                    Risk_Level,
                    Consistency_Result,
                    Prediction_Time
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """

        values=(transaction_id,transaction_amount,Anomaly_Score,LOF_Score,
                Anomaly_Label,LOF_Label,Risk_Level,Consistency_Result,datetime.now())
        cursor.execute(query,values)
        conn.commit()

        st.success("Data Saved Successfully")



main()

