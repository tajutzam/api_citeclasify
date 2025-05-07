from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import json  # pastikan sudah di-import di atas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import fitz
import re
from scopus import search_scopus,enrich_with_metrics
import time

from dotenv import load_dotenv
import os

import mysql.connector


load_dotenv()

API_KEY = os.getenv('TOKEN')


db = mysql.connector.connect(
    host=os.getenv('DB_HOST'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASS'),
    database=os.getenv('DB_NAME')
)
cursor = db.cursor()

app = Flask(__name__)

CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000"])


# Load model SVM dan TF-IDF Vectorizer
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("svm_model_new.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

# Load dataset
DATASET_PATH = "dataset/data_test.csv"
CONF_MATRIX_IMG_PATH = "upload/"


def extract_text_from_pdf(pdf_path):
    """Menangani ekstraksi teks dari file PDF"""
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_citation_sentences(text):
    """Menangani ekstraksi kalimat yang mengandung sitasi"""
    citation_patterns = [
        r'\[[0-9,\s]+\]',                            # [1], [2, 3]
        r'\([A-Z][a-z]+ et al\., \d{4}\)',           # (Smith et al., 2020)
        r'\([A-Z][a-z]+, \d{4}\)',                   # (Smith, 2020)
        r'\([A-Z][a-z]+ & [A-Z][a-z]+, \d{4}\)',     # (Smith & Johnson, 2020)
        r'\([A-Z][a-z]+, [A-Z][a-z]+, & [A-Z][a-z]+, \d{4}\)',  # (Smith, Johnson, & Lee, 2022)
        r'\([A-Z][a-z]+ et al\., \d{4}\)',           # (Smith et al., 2020)
        r'[A-Z][a-z]+ \(\d{4}\)',                    # Smith (2020)
        r'[A-Z][a-z]+ and [A-Z][a-z]+ \(\d{4}\)',    # Smith and Johnson (2021)
        r'[A-Z][a-z]+ et al\. \(\d{4}\)',            # Smith et al. (2020)
    ]
    
    combined_pattern = '|'.join(citation_patterns)
    
    # Pisahkan teks menjadi kalimat
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Ambil kalimat yang mengandung sitasi dan memiliki >= 5 kata
    citation_sentences = [
        s for s in sentences if re.search(combined_pattern, s) and len(s.split()) >= 5
    ]
    
    return citation_sentences

import fitz  # PyMuPDF
import numpy as np

def highlight_citation_sentences(pdf_path, output_path, vectorizer, svm_model):
    doc = fitz.open(pdf_path)
    all_text = extract_text_from_pdf(pdf_path)
    citation_sentences = extract_citation_sentences(all_text)

    prediction_results = []

    for page in doc:
        for sentence in citation_sentences:

            if any(item["text"] == sentence for item in prediction_results):
                continue

            tfidf_input = vectorizer.transform([sentence])
            prediction = svm_model.predict(tfidf_input)[0]
            if isinstance(prediction, (np.int64, np.int32)):
                prediction = int(prediction)

            try:
                probability = svm_model.predict_proba(tfidf_input)[0]
                confidence = float(np.max(probability))
            except AttributeError:
                confidence = None

            prediction_results.append({
                "text": sentence.replace("\n" , ""),
                "label": prediction,
                "accuracy": confidence
            })

            areas = page.search_for(sentence)
            for rect in areas:
                highlight = page.add_highlight_annot(rect)

                if prediction == 1:
                    highlight.set_colors(stroke=(1, 1, 0))  # Kuning
                elif prediction == 0:
                    highlight.set_colors(stroke=(1, 0, 0))  # Merah
                elif prediction == 2:
                    highlight.set_colors(stroke=(0, 1, 0))  # Hijau

                highlight.update()

    # Tambahkan keterangan di halaman pertama
    first_page = doc[0]
    legend_start_y = 50  # posisi awal dari atas
    spacing = 20

    legend_items = [
        {"label": "0 = background", "color": (1, 0, 0)},  # Merah
        {"label": "1 = method",     "color": (1, 1, 0)},  # Kuning
        {"label": "2 = result",     "color": (0, 1, 0)},  # Hijau
    ]

    for i, item in enumerate(legend_items):
        y = legend_start_y + i * spacing
        rect = fitz.Rect(50, y, 60, y + 10)
        first_page.draw_rect(rect, fill=item["color"], overlay=True)
        first_page.insert_text((65, y), item["label"], fontsize=10, color=(0, 0, 0))

    doc.save(output_path)
    doc.close()

    return {
        "output_path": output_path,
        "citation_sentences": prediction_results
    }



def extract_title_from_pdf(pdf_file, max_lines=2):
    """Menangani ekstraksi judul dari PDF berdasarkan teks tebal (bold), mengambil maksimal 3 baris"""
    doc = fitz.open(pdf_file)
    title = []
    
    # Looping melalui setiap halaman untuk mencari teks dengan format tebal (bold)
    for page in doc:
        blocks = page.get_text("dict")["blocks"]  # Mendapatkan teks dalam format dictionary
        
        for block in blocks:
            if block['type'] == 0:  # Teks biasa
                for line in block['lines']:
                    line_text = ""
                    for span in line['spans']:
                        # Periksa apakah gaya font adalah bold
                        if 'bold' in span['font'].lower():
                            # Ambil teks dari span yang tebal
                            line_text += span['text'] + " "  # Gabungkan teks dalam satu baris
                    
                    if line_text:
                        title.append(line_text.strip())  # Menambahkan baris ke judul

                    # Jika sudah mencapai batas jumlah baris, keluar dari loop
                    if len(title) >= max_lines:
                        break
            if len(title) >= max_lines:
                break
        if len(title) >= max_lines:
            break

    # Gabungkan baris-baris yang diambil dan kembalikan
    return " ".join(title) if title else "Title not found in the PDF"


@app.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """API untuk menerima file PDF dan mengembalikan kalimat dengan sitasi"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Simpan file PDF sementara
        pdf_path = f"./uploads/{file.filename}"
        file.save(pdf_path)

        # Ekstraksi teks dan sitasi dari file PDF
        text = extract_text_from_pdf(pdf_path)

        # Ganti \n dengan spasi di teks
        text = text.replace("\n", " ")
        
        timestamp = int(time.time())
        output_path = f'uploads/highlighted_output_{timestamp}.pdf'
        citation_sentences = highlight_citation_sentences(pdf_path, output_path , vectorizer, svm_model)

        title = request.form.get('title', None)
        
        if not title:  # Jika tidak ada title dalam request, ambil dari PDF
            title = extract_title_from_pdf(pdf_path)

        # Jika ada title dalam request, lakukan pencarian Scopus
        api_key = API_KEY
        query = f'TITLE("{title}")'

        df = search_scopus(query, api_key)

        search_results = enrich_with_metrics(df , api_key )


        return jsonify({
            "citation_sentences": citation_sentences['citation_sentences'],
            "search_results": search_results,
            "title": title,
            'pdf' : citation_sentences['output_path']
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/title-checker", methods=["POST"])
def check_judul():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Simpan file PDF sementara
        pdf_path = f"./uploads/{file.filename}"
        file.save(pdf_path)

        # Ekstrak judul dari PDF
        extracted_title = extract_title_from_pdf(pdf_path)

        return jsonify({
            "title": extracted_title
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint untuk prediksi satu teks
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.get_json()
        input_text = data.get("text", "")

        # Konversi teks ke TF-IDF
        tfidf_input = vectorizer.transform([input_text])

        # Prediksi label
        prediction = svm_model.predict(tfidf_input)[0]
        if isinstance(prediction, (np.int64, np.int32)):
            prediction = int(prediction)

        # Prediksi probabilitas jika model mendukung
        try:
            probability = svm_model.predict_proba(tfidf_input)[0].tolist()
        except AttributeError:
            probability = "Model tidak mendukung prediksi probabilitas"

        # Kembalikan hasil prediksi
        return jsonify({
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/upload/<filename>")
def get_uploaded_file(filename):
    return send_from_directory("upload", filename)


@app.route("/uploads/<filename>")
def get_pdf_highlight(filename):
    return send_from_directory("uploads", filename)


@app.route("/test", methods=["POST"])
def test_model():
    try:
        # Get test_size from the request, default is 1.0
        data = request.get_json()
        test_size = float(data.get("test_size", 1.0))  # Default to use all data

        # Load dataset
        df = pd.read_csv(DATASET_PATH)

        # Ensure the necessary columns exist
        if "clean_text_translated" not in df.columns or "label" not in df.columns:
            return jsonify({"error": "Dataset must contain 'clean_text_translated' and 'label' columns"}), 400

        # Convert labels to integers
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])

        # Use the entire dataset if test_size == 1.0, otherwise, use a portion
        if test_size == 1.0:
            X_test = df["clean_text_translated"]
            y_test = df["label"]
        else:
            _, X_test, _, y_test = train_test_split(
                df["clean_text_translated"], df["label"], test_size=test_size, random_state=42
            )

        # Convert text to TF-IDF
        X_test_tfidf = vectorizer.transform(X_test)

        # Predict with the existing model
        y_pred = svm_model.predict(X_test_tfidf)

        # Try predicting probabilities if the model supports it
        try:
            y_proba = svm_model.predict_proba(X_test_tfidf).tolist()
        except AttributeError:
            y_proba = "Model does not support probability predictions"

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # 🔥 Plot and save the confusion matrix as an image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf_matrix_img_filename = f"confusion_matrix_{timestamp}.png"
        conf_matrix_img_path = os.path.join(CONF_MATRIX_IMG_PATH, conf_matrix_img_filename)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(conf_matrix_img_path)  # Save the confusion matrix as an image
        plt.close()

        # Prepare predictions
        predictions = [
            {"text": text, "actual": int(actual), "predicted": int(pred), "probability": proba}
            for text, actual, pred, proba in zip(X_test, y_test, y_pred, y_proba)
        ] if isinstance(y_proba, list) else [
            {"text": text, "actual": int(actual), "predicted": int(pred), "probability": "N/A"}
            for text, actual, pred in zip(X_test, y_test, y_pred)
        ]

        plt.close()
        # Simpan ke tabel classification_reports
        sql = """
            INSERT INTO classification_reports 
            (accuracy, classification_report, confusion_matrix, test_size, total_data, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        """
        val = (
            accuracy,
            json.dumps(class_report),
            conf_matrix_img_path,
            test_size,
            len(predictions)
        )
        cursor.execute(sql, val)
        
        db.commit()

        # Return the results
        return jsonify({
            "test_size": test_size,
            "total_data": len(predictions),
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix_image_path": conf_matrix_img_path  # Return the image path
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
    
@app.route("/get_data", methods=["GET"])
def get_data():
    try:
        # Load dataset
        df = pd.read_csv(DATASET_PATH)

        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0" , 'processed_text' , 'translated'], inplace=True)

        # Konversi DataFrame ke list of dicts
        data = df.to_dict(orient="records")

        # Kembalikan data CSV dalam format JSON
        return jsonify({'data' : data})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
