import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from tensorflow.keras.models import load_model

# Load models from local files
svm_model = pickle.load(open("svm_model.pkl", "rb"))
xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

rnn_model = load_model("rnn_model.h5")
lstm_model = load_model("lstm_model.h5")
gru_model = load_model("gru_model.h5")
bi_rnn_model = load_model("bi_rnn_model.h5")
bi_lstm_model = load_model("bi_lstm_model.h5")
bi_gru_model = load_model("bi_gru_model.h5")

le = pickle.load(open("label_encoder.pkl", "rb"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# def prepare_input(question, answer, response):
#     q_emb = embedder.encode(question)
#     a_emb = embedder.encode(answer)
#     r_emb = embedder.encode(response)
#     cos_qr = cosine_similarity([q_emb], [r_emb])[0][0]
#     cos_ar = cosine_similarity([a_emb], [r_emb])[0][0]

#     flat_input = np.hstack([q_emb, a_emb, r_emb, [cos_qr], [cos_ar]])

#     # For classical ML models (SVM, XGBoost)
#     input_svm = flat_input.reshape(1, -1)
#     dmatrix_input = xgb.DMatrix(input_svm)

#     # For deep learning models (RNNs)
#     desired_len = 386
#     if flat_input.shape[0] > desired_len:
#         flat_input = flat_input[:desired_len]
#     else:
#         flat_input = np.pad(flat_input, (0, desired_len - flat_input.shape[0]))
#     input_rnn = flat_input.reshape(1, desired_len, 1)

#     return input_svm, input_rnn, dmatrix_input

def prepare_input(question, answer, response):
    q_emb = embedder.encode(question)
    a_emb = embedder.encode(answer)
    r_emb = embedder.encode(response)
    cos_qr = cosine_similarity([q_emb], [r_emb])[0][0]
    cos_ar = cosine_similarity([a_emb], [r_emb])[0][0]

    flat_input = np.hstack([q_emb, a_emb, r_emb, [cos_qr], [cos_ar]])

    desired_len = 386  # match SVM trained feature size

    if flat_input.shape[0] > desired_len:
        flat_input = flat_input[:desired_len]
    else:
        flat_input = np.pad(flat_input, (0, desired_len - flat_input.shape[0]))

    input_svm = flat_input.reshape(1, -1)
    dmatrix_input = xgb.DMatrix(input_svm)
    input_rnn = flat_input.reshape(1, desired_len, 1)

    return input_svm, input_rnn, dmatrix_input


def predict_from_input(question, answer, response):
    input_svm, input_rnn, input_xgb = prepare_input(question, answer, response)

    # Average prediction probs from all DL models
    probs_nn = (
        rnn_model.predict(input_rnn) +
        lstm_model.predict(input_rnn) +
        gru_model.predict(input_rnn) +
        bi_rnn_model.predict(input_rnn) +
        bi_lstm_model.predict(input_rnn) +
        bi_gru_model.predict(input_rnn)
    ) / 6

    # SVM prediction probabilities
    svm_probs = svm_model.predict_proba(input_svm)

    # XGBoost prediction probabilities (shape: (num_classes,))
    xgb_preds = xgb_model.predict(input_xgb)
    if xgb_preds.ndim == 1 or xgb_preds.shape[0] == 1:
        # Convert single-class prediction to one-hot probabilities if needed
        xgb_probs = np.eye(len(le.classes_))[xgb_preds.astype(int).flatten()]
    else:
        xgb_probs = xgb_preds

    # Average all model probabilities
    final_probs = (probs_nn + svm_probs + xgb_probs) / 3

    predicted_class = le.inverse_transform([np.argmax(final_probs, axis=1)[0]])[0]

    return predicted_class, final_probs[0]

# Streamlit UI starts here
st.title("🧠 Student Response Classification")

question = st.text_input("Enter the Question:")
answer = st.text_input("Enter the Correct Answer:")
response = st.text_area("Enter the Student's Response:")

if st.button("Predict"):
    if question and answer and response:
        pred_class, prob_dist = predict_from_input(question, answer, response)
        st.success(f"📌 Predicted Label: **{pred_class}**")

        st.subheader("Prediction Probabilities:")
        for cls, prob in zip(le.classes_, prob_dist):
            st.write(f"{cls}: {prob:.2%}")
    else:
        st.warning("Please enter question, answer, and response to predict.")
