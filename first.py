# Import necessary libraries
import streamlit as st
import pandas as pd
import opendatasets as od
import altair as alt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


# Define the main function
def main():
    uploaded_file = st.file_uploader(
        "Choose a CSV file from the nab folder", type="csv"
    )
    data = None
    # read the file as a pandas dataframe
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Pre-Processing",
            "EDA",
            "Isolation Forest",
            "LOF",
            "One Class SVM",
            "Evaluation Metric",
            "Resources Related To Project",
        ],
    )

    # Define behavior for "Home" page
    if page == "Home":
        st.title("MAJOR PROJECT - ANOMALY DETECTION")
        st.subheader("Made by Manas Katara - 2002139")
        st.subheader(
            "From Computer science Branch of DayalBagh Educational Institute, Agra"
        )
        st.subheader("Under Supervision of Dr. Rajeev Kumar Chauhan")

        # Project overview and details
        st.markdown(
            """
            ## Overview
            ### Project Details
            In this project, we use NAB-dataset, a benchmark for evaluating anomaly detection algorithms.
            There are 58 timeseries data from various sources.

            #### Real data
            - realAWSCloudwatch
            - realAdExchange
            - realKnownCause
            - realTraffic
            - realTweets

            #### Artificial data
            - artificialNoAnomaly
            - artificialWithAnomaly 

            In these datasets, we analyze 'machine_temperature_system_failure' from realKnownCause.

            ### Goal of this project
            - Practice data pre-processing technique
            - Practice EDA technique for time-series data
            - Practice visualization techniques
            - Practice anomaly detection (Isolation Forest)
            - Practice improving model interpretability (SHAP)
            """
        )

        # Load and analyze data button
        st.markdown("## Data Analysis Section")
        if st.button("Load and Analyze Data"):
            st.write("Loaded File Data Starting five Row")
            st.dataframe(data.head(5))

    # Define behavior for "Pre-Processing" page
    elif page == "Pre-Processing":
        st.header("Data Pre-Processing")
        st.subheader("Categorising Data By Year, Months, Day, Hour, and Minutes")

        # Define anomaly points and categorize data
        anomaly_points = [
            ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
            ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
            ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
            ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"],
        ]
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["Anomaly"] = 0

        # Mark anomaly points
        for start, end in anomaly_points:
            data.loc[
                ((data["timestamp"] >= start) & (data["timestamp"] <= end)),
                "Anomaly",
            ] = 1

        # Extract time components
        data["year"] = data["timestamp"].apply(lambda x: x.year)
        data["month"] = data["timestamp"].apply(lambda x: x.month)
        data["day"] = data["timestamp"].apply(lambda x: x.day)
        data["hour"] = data["timestamp"].apply(lambda x: x.hour)
        data["minute"] = data["timestamp"].apply(lambda x: x.minute)

        # Set timestamp as the index
        data.index = data["timestamp"]
        data.drop(["timestamp"], axis=1, inplace=True)

        # Display a slider to select the number of rows
        num_rows = st.slider("Select the number of rows", 5, 200)
        st.dataframe(data.head(num_rows))

    # Define behavior for "EDA" page
    elif page == "EDA":
        st.header("Exploratory Data Analysis")

        # Define the data variable here

        st.markdown("## Temperature Analysis")
        if st.button("Show Temperature Analysis"):
            if data is not None:
                st.dataframe(data, column_order=[])
            else:
                st.warning("Please Upload a CSV File to perform EDA")

            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["year"] = data["timestamp"].apply(lambda x: x.year)
            data["month"] = data["timestamp"].apply(lambda x: x.month)

            # Reset the index before using it with Altair
            data_grouped_count = (
                data.groupby(["year", "month"])["value"].count().reset_index()
            )

            data_grouped_mean = (
                data.groupby(["year", "month"])["value"].mean().reset_index()
            )

            year_maxmin = (
                data.groupby(["year", "month"])
                .agg({"value": ["min", "max"]})
                .reset_index()
            )
            year_maxmin.columns = [
                "_".join(col).strip() for col in year_maxmin.columns.values
            ]

            # Altair bar chart
            chart_count = (
                alt.Chart(data_grouped_count)
                .mark_bar()
                .encode(
                    x="year:N",
                    y="value:Q",
                    color=alt.Color("year:N", scale=alt.Scale(scheme="turbo")),
                    tooltip=["year:N", "value:Q"],
                )
                .properties(title="Year/Month Count")
            )

            chart_mean = (
                alt.Chart(data_grouped_mean)
                .mark_bar()
                .encode(
                    x="year:N",
                    y="value:Q",
                    color=alt.Color("year:N", scale=alt.Scale(scheme="plasma")),
                    tooltip=["year:N", "value:Q"],
                )
                .properties(title="Year/Month Mean Temperature")
            )

            chart_max_temp = (
                alt.Chart(year_maxmin)
                .mark_bar()
                .encode(
                    x="year_:N",
                    y="value_max:Q",
                    color=alt.Color("year_:N", scale=alt.Scale(scheme="sinebow")),
                    tooltip=["year_:N", "value_max:Q"],
                )
                .properties(title="Year/Month Max Temperature")
            )

            chart_min_temp = (
                alt.Chart(year_maxmin)
                .mark_bar()
                .encode(
                    x="year_:N",
                    y="value_min:Q",
                    color=alt.Color("year_:N", scale=alt.Scale(scheme="rainbow")),
                    tooltip=["year_:N", "value_min:Q"],
                )
                .properties(title="Year/Month Min Temperature")
            )

            # Display Altair charts using st.altair_chart
            st.altair_chart(chart_count, use_container_width=True)
            st.altair_chart(chart_mean, use_container_width=True)
            st.altair_chart(chart_max_temp, use_container_width=True)
            st.altair_chart(chart_min_temp, use_container_width=True)

        if st.button("Show Temperature Distribution"):
            if data is not None:
                # Temperature Distribution using Altair
                distribution_area_chart = (
                    alt.Chart(data)
                    .mark_area(
                        interpolate="basis",
                        line={"color": "skyblue"},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="blue", offset=0),
                                alt.GradientStop(color="red", offset=1),
                            ],
                            x1=1,
                            x2=1,
                            y1=1,
                            y2=0,
                        ),
                    )
                    .encode(
                        alt.X("value:Q", bin=alt.Bin(step=1), title="Temperature"),
                        alt.Y("count():Q", title="Density"),
                        tooltip=["count()"],
                    )
                    .properties(
                        title="Temperature Distribution (Area Chart)",
                        width=700,
                        height=500,
                    )
                )
                st.altair_chart(distribution_area_chart, use_container_width=True)

    # Define behavior for "Isolation Forest" page
    elif page == "Isolation Forest":
        st.header("Isolation Forest Anomalies Observation")

        # Load data

        # Isolation Forest
        iforest_model = IsolationForest(
            n_estimators=300, contamination=0.1, max_samples=700
        )
        iforest_ret = iforest_model.fit_predict(data["value"].values.reshape(-1, 1))
        iforest_df = pd.DataFrame()
        iforest_df["index"] = data.index
        iforest_df["value"] = data["value"]
        iforest_df["anomaly"] = [1 if i == -1 else 0 for i in iforest_ret]

        # Display Isolation Forest graph button
        if st.button("Show Isolation Forest Graph"):
            # Altair Plot
            chart = (
                alt.Chart(iforest_df)
                .mark_circle(size=60)
                .encode(
                    x="index:T",
                    y="value:Q",
                    color=alt.condition(
                        alt.datum.anomaly == 1,
                        alt.value("orange"),
                        alt.value("skyblue"),
                    ),
                    tooltip=["index:T", "value:Q", "anomaly:N"],
                )
                .properties(title="Isolation Forest Anomalies Observation")
            )

            # Display Altair chart using st.altair_chart
            st.altair_chart(chart, use_container_width=True)

        st.header("Isolation Forest Evaluation Metric Observation")
        if st.button("Show ROC-AUC and F1 score And Curve"):
            anomaly_points = [
                ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
                ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
                ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
                ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"],
            ]
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["Anomaly"] = 0

            for start, end in anomaly_points:
                data.loc[
                    ((data["timestamp"] >= start) & (data["timestamp"] <= end)),
                    "Anomaly",
                ] = 1

            iforest_f1 = f1_score(data["Anomaly"], iforest_df["anomaly"])
            st.write(f"Isolation Forest F1 Score: {iforest_f1}")
            iforest_auc_roc = roc_auc_score(data["Anomaly"], iforest_df["anomaly"])
            st.write(f"Isolation Forest AUC-ROC Score: {iforest_auc_roc}")
            fpr, tpr, thresholds = roc_curve(data["Anomaly"], iforest_df["anomaly"])
            roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
            roc_curve_plot = (
                alt.Chart(roc_data)
                .mark_line(color="orange")
                .encode(x="FPR", y="TPR", tooltip=["FPR", "TPR"])
                .properties(
                    width=600,
                    height=400,
                    title=f"Receiver Operating Characteristic (ROC) Curve\nAUC = {iforest_auc_roc:.2f}",
                )
            )

            diagonal_line = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(color="gray", strokeDash=[3, 3])
                .encode(x="x", y="y")
            )

            roc_chart = (
                (roc_curve_plot + diagonal_line)
                .configure_axis(labelFontSize=12, titleFontSize=14)
                .configure_legend(titleFontSize=14, labelFontSize=12)
                .configure_title(fontSize=16)
            )

            st.altair_chart(roc_chart, use_container_width=True)
            st.write("TERMS USED")
            st.write("FPR = False Positive Rate")
            st.write("TPR = True Positive Rate")

    # Define behavior for "LOF" page
    elif page == "LOF":
        # Add LOF page content here
        st.header("Local Outlier Factor (LOF) Anomalies Observation")

        # LOF model
        lof_model = LocalOutlierFactor(
            n_neighbors=500,
            contamination=0.07,
        )
        lof_ret = lof_model.fit_predict(data["value"].values.reshape(-1, 1))
        lof_df = pd.DataFrame()
        lof_df["index"] = data.index
        lof_df["value"] = data["value"]
        lof_df["anomaly"] = [1 if i == -1 else 0 for i in lof_ret]

        if st.button("Show LOF Graph"):
            # Altair Plot for LOF
            chart_lof = (
                alt.Chart(lof_df)
                .mark_circle(size=60)
                .encode(
                    x="index:T",
                    y="value:Q",
                    color=alt.condition(
                        alt.datum.anomaly == 1, alt.value("red"), alt.value("skyblue")
                    ),
                    tooltip=["index:T", "value:Q", "anomaly:N"],
                )
                .properties(title="Local Outlier Factor (LOF) Anomalies Observation")
            )

            # Display Altair chart using st.altair_chart
            st.altair_chart(chart_lof, use_container_width=True)

        st.header("LOF Evaluation Metric Observation")
        if st.button("Show LOF ROC-AUC and F1 score And Curve"):
            anomaly_points = [
                ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
                ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
                ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
                ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"],
            ]
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["Anomaly"] = 0
            for start, end in anomaly_points:
                data.loc[
                    ((data["timestamp"] >= start) & (data["timestamp"] <= end)),
                    "Anomaly",
                ] = 1

            lof_f1 = f1_score(data["Anomaly"], lof_df["anomaly"])
            st.write(f"LOF F1 Score: {lof_f1}")
            lof_auc_roc = roc_auc_score(data["Anomaly"], lof_df["anomaly"])
            st.write(f"LOF AUC-ROC Score: {lof_auc_roc}")
            fpr, tpr, thresholds = roc_curve(data["Anomaly"], lof_df["anomaly"])
            roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
            roc_curve_plot = (
                alt.Chart(roc_data)
                .mark_line(color="red")
                .encode(x="FPR", y="TPR", tooltip=["FPR", "TPR"])
                .properties(
                    width=600,
                    height=400,
                    title=f"Receiver Operating Characteristic (ROC) Curve\nAUC = {lof_auc_roc:.2f}",
                )
            )

            diagonal_line = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(color="gray", strokeDash=[3, 3])
                .encode(x="x", y="y")
            )

            roc_chart = (
                (roc_curve_plot + diagonal_line)
                .configure_axis(labelFontSize=12, titleFontSize=14)
                .configure_legend(titleFontSize=14, labelFontSize=12)
                .configure_title(fontSize=16)
            )

            st.altair_chart(roc_chart, use_container_width=True)
            st.write("TERMS USED")
            st.write("FPR = False Positive Rate")
            st.write("TPR = True Positive Rate")

    # Define behavior for "One Class SVM" page
    elif page == "One Class SVM":
        # Add One Class SVM page content here
        st.header("One-Class SVM Anomalies Observation")
        if data is not None:
            # LOF model
            ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel="rbf")
            ocsvm_ret = ocsvm_model.fit_predict(data["value"].values.reshape(-1, 1))
            ocsvm_df = pd.DataFrame()
            ocsvm_df["index"] = data.index
            ocsvm_df["value"] = data["value"]
            ocsvm_df["anomaly"] = [1 if i == -1 else 0 for i in ocsvm_ret]

            if st.button("Show LOF Graph"):
                # Altair Plot for LOF
                chart_ocsvm = (
                    alt.Chart(ocsvm_df)
                    .mark_circle(size=60)
                    .encode(
                        x="index:T",
                        y="value:Q",
                        color=alt.condition(
                            alt.datum.anomaly == 1,
                            alt.value("pink"),
                            alt.value("skyblue"),
                        ),
                        tooltip=["index:T", "value:Q", "anomaly:N"],
                    )
                    .properties(title="One Class SVM (OCSVM) Anomalies Observation")
                )

            # Display Altair chart using st.altair_chart
            st.altair_chart(chart_ocsvm, use_container_width=True)

        st.header("OCSVM Evaluation Metric Observation")
        if st.button("Show OCSVM ROC-AUC and F1 score And Curve"):
            anomaly_points = [
                ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
                ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
                ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
                ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"],
            ]
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["Anomaly"] = 0
            for start, end in anomaly_points:
                data.loc[
                    ((data["timestamp"] >= start) & (data["timestamp"] <= end)),
                    "Anomaly",
                ] = 1

            ocsvm_f1 = f1_score(data["Anomaly"], ocsvm_df["anomaly"])
            st.write(f"LOF F1 Score: {ocsvm_f1}")
            ocsvm_auc_roc = roc_auc_score(data["Anomaly"], ocsvm_df["anomaly"])
            st.write(f"LOF AUC-ROC Score: {ocsvm_auc_roc}")
            fpr, tpr, thresholds = roc_curve(data["Anomaly"], ocsvm_df["anomaly"])
            roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Thresholds": thresholds})
            roc_curve_plot = (
                alt.Chart(roc_data)
                .mark_line(color="red")
                .encode(x="FPR", y="TPR", tooltip=["FPR", "TPR"])
                .properties(
                    width=600,
                    height=400,
                    title=f"Receiver Operating Characteristic (ROC) Curve\nAUC = {ocsvm_auc_roc:.2f}",
                )
            )

            diagonal_line = (
                alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
                .mark_line(color="gray", strokeDash=[3, 3])
                .encode(x="x", y="y")
            )

            roc_chart = (
                (roc_curve_plot + diagonal_line)
                .configure_axis(labelFontSize=12, titleFontSize=14)
                .configure_legend(titleFontSize=14, labelFontSize=12)
                .configure_title(fontSize=16)
            )

            st.altair_chart(roc_chart, use_container_width=True)
            st.write("TERMS USED")
            st.write("FPR = False Positive Rate")
            st.write("TPR = True Positive Rate")

    elif page == "Evaluation Metric":
        st.header("Evaluation Matrices Results for Different Models")

        # Check if data is loaded
        if data is not None:
            # Define anomaly points
            anomaly_points = [
                ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
                ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
                ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
                ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"],
            ]
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["Anomaly"] = 0

            # Mark anomaly points
            for start, end in anomaly_points:
                data.loc[
                    ((data["timestamp"] >= start) & (data["timestamp"] <= end)),
                    "Anomaly",
                ] = 1

            # Define models
            models = ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]

            # Initialize result dictionaries
            f1_scores = {}
            roc_scores = {}

            # Evaluate each model
            for model_name in models:
                if model_name == "Isolation Forest":
                    model = IsolationForest(
                        n_estimators=300, contamination=0.1, max_samples=700
                    )
                    model_result = model.fit_predict(
                        data["value"].values.reshape(-1, 1)
                    )
                elif model_name == "Local Outlier Factor":
                    model = LocalOutlierFactor(n_neighbors=500, contamination=0.07)
                    model_result = model.fit_predict(
                        data["value"].values.reshape(-1, 1)
                    )
                elif model_name == "One-Class SVM":
                    model = OneClassSVM(nu=0.2, gamma=0.001, kernel="rbf")
                    model_result = model.fit_predict(
                        data["value"].values.reshape(-1, 1)
                    )

                # Calculate F1 score and ROC score
                f1 = f1_score(
                    data["Anomaly"], [1 if i == -1 else 0 for i in model_result]
                )
                roc = roc_auc_score(
                    data["Anomaly"], [1 if i == -1 else 0 for i in model_result]
                )

                # Save results to dictionaries
                f1_scores[model_name] = f1
                roc_scores[model_name] = roc

            # Create a DataFrame to display results
            results_df = pd.DataFrame(
                {
                    "Model": models,
                    "F1 Score": list(f1_scores.values()),
                    "ROC AUC Score": list(roc_scores.values()),
                }
            )

            # Display the table
            st.table(results_df)
        else:
            st.warning("Please Upload a CSV File to evaluate models.")

    elif page == "Resources Related To Project":
        st.markdown(
            """
            ## Resource and Backend Page
           Here you can find the complete project code.
           and the all the neccessory information regarding the project implemeted"""
        )
        if st.button("Download Project Report"):
            # Replace 'your_url_here' with the actual URL of your Google Drive file
            st.markdown(
                '<a href="" target="_blank">Download Report</a>',
                unsafe_allow_html=True,
            )

        # Second button for showing a webpage
        if st.button("Show Webpage"):
            st.markdown(
                """
                #### WEB PAGE DETAILS
                here is the webpage in which one more model is implemented which is "MICROSOFT ANOMALY DETECTION SERVICE"
                This webpage also consists of all the EDA that we have done in the project with more rigorous EDA.
                And here you can find the backend code of the project in PYTHON.
            """
            )
            components.iframe(
                "https://anamolydetection.000webhostapp.com",
                height=1200,
                width=800,
                scrolling=True,
            )


# Execute main function if the script is run directly
if __name__ == "__main__":
    main()
