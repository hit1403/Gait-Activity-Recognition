import pywt
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


st.title("**GAIT ACTIVITY RECOGNITION**")
st.header("Upload the dataset")
# Load the dataset from a local Excel file
uploaded_file = st.file_uploader("Upload a dataset (Excel file)", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.header("Displaying the dataset")
    st.dataframe(df)



    # Split your data into features (X) and labels (Y)
    X = df.iloc[1:, 0:18]
    Y = df.iloc[1:, 18]
    """
    # Convert labels to categorical values
    lab = preprocessing.LabelEncoder()
    Y = lab.fit_transform(Y)
    """

    # Normalize features using MinMaxScaler
    min_range = 0
    max_range = 5
    scaler = MinMaxScaler(feature_range=(min_range, max_range))

    # Normalize only the numeric columns in X
    numeric_columns = X.select_dtypes(include=['number']).columns
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    print("------------after normalising-------: \n",df)
    # Perform CWT and generate new frequency features
    def perform_cwt(signal, scales):
        cwt_result, _ = pywt.cwt(signal, scales, 'morl')
        return np.abs(cwt_result)

    # Parameters for CWT
    scales = np.arange(1, 28)  # Create 27 scales for the CWT

    # Apply CWT to each sensor signal
    cwt_features = []
    for sensor_column in X.columns:
        cwt_data = perform_cwt(X[sensor_column], scales)
        cwt_features.append(cwt_data)

    # Combine CWT features into a single matrix
    cwt_features = np.vstack(cwt_features).transpose()

    # Add CWT features to the existing feature matrix
    X = pd.DataFrame(np.concatenate([X, cwt_features], axis=1))

    # Split the data into training and testing sets
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, train_size=0.8, random_state=42)

    # Define and train an Autoencoder for feature reduction
    input_dim = X_TRAIN.shape[1]
    encoding_dim = 10  # You can adjust this as needed

    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
    decoder = tf.keras.layers.Dense(input_dim, activation="relu")(encoder)
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X_TRAIN, X_TRAIN, epochs=50, batch_size=64, shuffle=True, validation_data=(X_TEST, X_TEST))

    # Use the encoder part of the autoencoder for feature reduction
    encoder_model = tf.keras.models.Model(inputs=input_layer, outputs=encoder)
    X_TRAIN_encoded = encoder_model.predict(X_TRAIN)
    X_TEST_encoded = encoder_model.predict(X_TEST)



    # Define the SVM model
    svm_params = {
        'kernel': 'linear',  # Linear kernel
        'decision_function_shape': 'ovo',  # OneVsOne
        'C': 1.0,  # You can optimize this with Bayesian optimization
        'max_iter': 7,  # Max epochs
        'probability': True,  # To calculate class prior probabilities
    }

    model_01 = SVC(**svm_params)

    # perform Bayesian optimization of the C parameter here

    # Spliting the data into KFold for cross-validation (K=4)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # Use GridSearchCV to find the best parameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # You can add more values to search
        'kernel': ['linear', 'rbf', 'poly'],  # You can add other kernels
    }

    grid_search = GridSearchCV(estimator=model_01, param_grid=param_grid, scoring='accuracy', cv=kf)
    #hitttttttttttttttttt
    grid_search.fit(X_TRAIN, Y_TRAIN)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    model_01.set_params(**best_params)

    # Train the SVM model
    model_01.fit(X_TRAIN, Y_TRAIN)

    # Predictions
    PREDICTIONS_01 = model_01.predict(X_TEST)


    # Calculate precision
    precision = precision_score(Y_TEST, PREDICTIONS_01,average = 'weighted')

    # Calculate recall (sensitivity)
    recall = recall_score(Y_TEST, PREDICTIONS_01,average = 'macro')

    # Calculate F1-score
    f1 = f1_score(Y_TEST, PREDICTIONS_01,average = 'weighted')

    # Calculate accuracy
    accuracy = accuracy_score(Y_TEST, PREDICTIONS_01)

    # Calculate standard deviation
    std_dev = np.std(PREDICTIONS_01)

    yte = pd.Series(Y_TEST)
    PREDICTIONS_01 = pd.Series(PREDICTIONS_01)

    st.subheader("Label indication:")
    st.write("0 -> STANDING STILL")
    st.write("1 -> WALKING")
    st.write("2 -> RUNNING")
    st.write("3 -> DESCENDING STAIRS")
    st.write("4 -> ASCENDING STAIRS")
    st.subheader("For testing data")
    activity_counts = yte.value_counts()
    fig, ax = plt.subplots()
    ax.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    st.subheader("For predicted data")
    activity_counts = PREDICTIONS_01.value_counts()
    fig, ax = plt.subplots()
    ax.pie(activity_counts, labels=activity_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)


    # Display evaluation metrics
    st.header("Model Evaluation Metrics")
    st.write("**Precision**\t\t\t: ",precision)
    st.write("**Recall (Sensitivity)**\t\t\t:",recall)
    st.write("**F1 Score**\t\t\t:",f1)
    st.write("**Accuracy**\t\t\t:",accuracy)
    st.write("**Standard deviation**\t\t\t:",std_dev)


