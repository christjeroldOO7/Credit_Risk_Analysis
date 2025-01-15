def process_df(xdf: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    st.write("Initial Data Preview:")
    st.dataframe(xdf)  # Display initial data in Streamlit

    xdf = normalize_df(xdf)
    st.write("After Normalization:")
    st.dataframe(xdf)

    xdf = labels_encode(xdf)
    st.write("After Label Encoding:")
    st.dataframe(xdf)

    xdf = apply_smote(xdf)
    st.write("After SMOTE Resampling:")
    st.dataframe(xdf)

    splitted = split_ml_df(xdf, 0.2)
    keys = ["X_train", "X_test", "y_train", "y_test"]
    return {k: v for [k, v] in zip(keys, splitted)}

@st.cache_resource
def xgbclassifier_model(X_train, y_train) -> XGBClassifier:
    model = XGBClassifier(**BEST_PARAMS, random_state=RANDOM_STATE)

    try:
        st.write("Starting model training...")
        model.fit(X_train, y_train, eval_metric="logloss", verbose=True)
        st.success("Model training completed successfully!")
    except ValueError as ve:
        st.error(f"ValueError during training: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None

    return model

def get_label_encoding(coll: Dict[str, Dict[str, int]], label: str, key: str) -> int:
    return coll[label][key]

def scale_for_pred(
    coll: Dict[str, MinMaxScaler], label: str, val: Union[int, float]
) -> float:
    scaler: MinMaxScaler = coll[label]
    val = [[val]]
    return scaler.transform(val)[0][0]

def transform_for_pred(
    coll_num: Dict[str, MinMaxScaler],
    coll_cat: Dict[str, Dict[str, int]],
    form: Dict[str, Union[str, int, float]],
) -> pd.DataFrame:
    for k, v in form.items():
        if k in coll_cat:  # Categorical feature
            form[k] = get_label_encoding(coll_cat, k, v)
        elif k in coll_num:  # Numerical feature
            form[k] = scale_for_pred(coll_num, k, v)

    return pd.DataFrame(data=[form])

# Streamlit App
st.title("XGBoost Classifier with Preprocessing")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded File Preview:")
    st.dataframe(data.head())

    if "class" not in data.columns:
        st.error("The dataset must contain a 'class' column.")
    else:
        processed_data = process_df(data)
        X_train, X_test, y_train, y_test = (
            processed_data["X_train"],
            processed_data["X_test"],
            processed_data["y_train"],
            processed_data["y_test"],
        )

        st.write("Training and Testing Data Shapes:")
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"X_test shape: {X_test.shape}")
        st.write(f"y_train shape: {y_train.shape}")
        st.write(f"y_test shape: {y_test.shape}")

        if st.button("Train Model"):
            st.write("Training the model...")
            model = xgbclassifier_model(X_train, y_train)
            if model:
                accuracy = model.score(X_test, y_test)
                st.success(f"Model Accuracy: {accuracy:.2f}")
else:
    st.info("Please upload a dataset to begin.")
