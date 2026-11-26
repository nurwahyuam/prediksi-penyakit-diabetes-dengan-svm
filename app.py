"""Aplikasi web Flask untuk demo klasifikasi status diabetes (SVM, BRFSS 2015).

Menjalankan:  python app.py   lalu buka http://127.0.0.1:5000
Aplikasi ini ALAT PENELITIAN/EDUKASI, bukan alat diagnosis medis.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CANDIDATES = [
    os.environ.get("MODEL_PATH"),
    os.path.join(BASE_DIR, "model", "svm_diabetes_app.joblib"),
    os.path.join(BASE_DIR, "svm_diabetes_app.joblib"),
    os.path.join(BASE_DIR, "..", "svm_diabetes_app.joblib"),
]

YN = [(0, "Tidak"), (1, "Ya")]

# Deskripsi fitur: nama kolom dataset (harus persis sama dengan saat pelatihan),
# label Indonesia, dan batas nilai. Arti kode mengikuti dokumentasi dataset/kodebook BRFSS;
# periksa kembali terhadap dokumentasi dataset yang Anda pakai.
FEATURES = [
    {"group": "Data diri", "name": "Sex", "label": "Jenis kelamin", "type": "select", "options": [(0, "Perempuan"), (1, "Laki-laki")]},
    {"group": "Data diri", "name": "Age", "label": "Kategori usia", "type": "select", "options": [
        (1, "18–24"), (2, "25–29"), (3, "30–34"), (4, "35–39"), (5, "40–44"), (6, "45–49"), (7, "50–54"),
        (8, "55–59"), (9, "60–64"), (10, "65–69"), (11, "70–74"), (12, "75–79"), (13, "80 tahun atau lebih")]},
    {"group": "Data diri", "name": "BMI", "label": "Indeks massa tubuh (BMI)", "type": "number", "min": 10, "max": 100, "step": 0.1,
     "help": "Berat badan (kg) dibagi tinggi badan (m) kuadrat."},
    {"group": "Kondisi kesehatan", "name": "HighBP", "label": "Pernah didiagnosis tekanan darah tinggi", "type": "select", "options": YN},
    {"group": "Kondisi kesehatan", "name": "HighChol", "label": "Pernah didiagnosis kolesterol tinggi", "type": "select", "options": YN},
    {"group": "Kondisi kesehatan", "name": "CholCheck", "label": "Pemeriksaan kolesterol dalam 5 tahun terakhir", "type": "select", "options": YN},
    {"group": "Kondisi kesehatan", "name": "Stroke", "label": "Pernah stroke", "type": "select", "options": YN},
    {"group": "Kondisi kesehatan", "name": "HeartDiseaseorAttack", "label": "Penyakit jantung koroner atau pernah serangan jantung", "type": "select", "options": YN},
    {"group": "Kondisi kesehatan", "name": "DiffWalk", "label": "Kesulitan berjalan atau naik tangga", "type": "select", "options": YN},
    {"group": "Kondisi kesehatan", "name": "GenHlth", "label": "Penilaian kesehatan umum", "type": "select", "options": [
        (1, "1 – Sangat baik"), (2, "2 – Baik"), (3, "3 – Cukup"), (4, "4 – Kurang"), (5, "5 – Buruk")]},
    {"group": "Kondisi kesehatan", "name": "MentHlth", "label": "Hari dengan kesehatan mental kurang baik (30 hari terakhir)", "type": "number", "min": 0, "max": 30, "step": 1},
    {"group": "Kondisi kesehatan", "name": "PhysHlth", "label": "Hari dengan kesehatan fisik kurang baik (30 hari terakhir)", "type": "number", "min": 0, "max": 30, "step": 1},
    {"group": "Gaya hidup", "name": "Smoker", "label": "Pernah merokok minimal 100 batang seumur hidup", "type": "select", "options": YN},
    {"group": "Gaya hidup", "name": "PhysActivity", "label": "Aktivitas fisik dalam 30 hari terakhir (di luar pekerjaan)", "type": "select", "options": YN},
    {"group": "Gaya hidup", "name": "Fruits", "label": "Makan buah minimal sekali sehari", "type": "select", "options": YN},
    {"group": "Gaya hidup", "name": "Veggies", "label": "Makan sayur minimal sekali sehari", "type": "select", "options": YN},
    {"group": "Gaya hidup", "name": "HvyAlcoholConsump", "label": "Peminum alkohol berat", "type": "select", "options": YN},
    {"group": "Layanan kesehatan dan sosial-ekonomi", "name": "AnyHealthcare", "label": "Memiliki jaminan atau akses layanan kesehatan", "type": "select", "options": YN},
    {"group": "Layanan kesehatan dan sosial-ekonomi", "name": "NoDocbcCost", "label": "Tidak berobat ke dokter dalam 12 bulan terakhir karena biaya", "type": "select", "options": YN},
    {"group": "Layanan kesehatan dan sosial-ekonomi", "name": "Education", "label": "Tingkat pendidikan", "type": "select", "options": [
        (1, "1 – Tidak sekolah atau TK"), (2, "2 – SD"), (3, "3 – SMP (belum lulus SMA)"),
        (4, "4 – Lulus SMA"), (5, "5 – Pernah kuliah atau pendidikan vokasi"), (6, "6 – Lulus perguruan tinggi")]},
    {"group": "Layanan kesehatan dan sosial-ekonomi", "name": "Income", "label": "Kategori pendapatan rumah tangga", "type": "select", "options": [
        (1, "1 – Terendah"), (2, "2"), (3, "3"), (4, "4"), (5, "5"), (6, "6"), (7, "7"), (8, "8 – Tertinggi")],
     "help": "Skala 1–8 dari BRFSS (kategori pendapatan dalam dolar AS); pilih tingkat relatif."},
]
GROUPS = []
for f in FEATURES:
    if f["group"] not in GROUPS:
        GROUPS.append(f["group"])
DEFAULTS = {"Sex": 0, "Age": 7, "BMI": 27, "HighBP": 0, "HighChol": 0, "CholCheck": 1, "Stroke": 0,
            "HeartDiseaseorAttack": 0, "DiffWalk": 0, "GenHlth": 3, "MentHlth": 0, "PhysHlth": 0,
            "Smoker": 0, "PhysActivity": 1, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
            "AnyHealthcare": 1, "NoDocbcCost": 0, "Education": 5, "Income": 6}

app = Flask(__name__)


def load_bundle():
    tried = []
    for path in MODEL_CANDIDATES:
        if not path:
            continue
        path = os.path.abspath(path)
        tried.append(path)
        if os.path.isfile(path):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    bundle = joblib.load(path)
                bundle["_path"] = path
                bundle["_warnings"] = [str(x.message)[:200] for x in w]
                return bundle, None
            except Exception as exc:  # berkas ada tetapi gagal dimuat
                return None, f"Berkas model ditemukan ({path}) tetapi gagal dimuat: {exc}. " \
                             "Pastikan versi scikit-learn sama dengan yang dipakai saat pelatihan."
    return None, "Berkas model svm_diabetes_app.joblib tidak ditemukan. Jalankan Bagian 14A pada notebook, " \
                 "lalu letakkan berkas itu di folder flask_app/model/. Lokasi yang dicari: " + "; ".join(tried)


BUNDLE, LOAD_ERROR = load_bundle()


def parse_inputs(values):
    """Validasi dan ubah input menjadi dict {nama_fitur: float}. Mengembalikan (row, errors)."""
    row, errors = {}, {}
    for f in FEATURES:
        raw = values.get(f["name"])
        if raw is None or str(raw).strip() == "":
            errors[f["name"]] = "Wajib diisi."
            continue
        try:
            val = float(str(raw).replace(",", "."))
        except ValueError:
            errors[f["name"]] = "Harus berupa angka."
            continue
        if not np.isfinite(val):
            errors[f["name"]] = "Nilai tidak valid."
            continue
        if f["type"] == "select":
            allowed = [float(o[0]) for o in f["options"]]
            if val not in allowed:
                errors[f["name"]] = "Pilihan tidak valid."
                continue
        else:
            if not (f["min"] <= val <= f["max"]):
                errors[f["name"]] = f"Harus antara {f['min']} dan {f['max']}."
                continue
        row[f["name"]] = val
    return row, errors


def predict_row(row):
    names = BUNDLE["feature_names"]
    missing = [n for n in names if n not in row]
    if missing:
        raise ValueError("Fitur model tidak ada pada formulir: " + ", ".join(missing))
    df = pd.DataFrame([row])[names]
    model = BUNDLE["model"]
    pred = int(model.predict(df)[0])
    class_names = BUNDLE["class_names"]
    labels = BUNDLE.get("labels") or list(range(len(class_names)))
    label = class_names[labels.index(pred)]
    scores = None
    if hasattr(model, "decision_function"):
        d = np.atleast_1d(np.asarray(model.decision_function(df))).reshape(1, -1)[0]
        if len(d) == len(class_names):
            lo, hi = float(d.min()), float(d.max())
            span = (hi - lo) or 1.0
            scores = [{"nama": n, "nilai": round(float(v), 3), "persen": round(100 * (float(v) - lo) / span)}
                      for n, v in zip(class_names, d)]
        elif len(d) == 1 and len(class_names) == 2:
            scores = [{"nama": f"Skor condong ke '{class_names[1]}' (positif) atau '{class_names[0]}' (negatif)",
                       "nilai": round(float(d[0]), 3), "persen": None}]
    return {"kelas": pred, "label": label, "skor": scores}


def model_info():
    if not BUNDLE:
        return None
    m = BUNDLE.get("metrics") or {}
    shown = []
    for key, text in [("Accuracy", "Accuracy"), ("Balanced_Acc", "Balanced accuracy"), ("Macro_F1", "Macro-F1")]:
        if key in m:
            shown.append((text, f"{m[key]:.3f}"))
    for key, val in m.items():
        if key.startswith("Recall_"):
            shown.append((f"Recall {key[7:].replace('_', ' ').lower()}", f"{val:.3f}"))
    return {"nama": BUNDLE.get("model_name", "-"), "task": BUNDLE.get("task", "-"), "metrik": shown,
            "peringatan": BUNDLE.get("_warnings", [])}


@app.route("/", methods=["GET", "POST"])
def index():
    values, errors, result = dict(DEFAULTS), {}, None
    if request.method == "POST":
        values = {f["name"]: request.form.get(f["name"], "") for f in FEATURES}
        if BUNDLE is None:
            errors["_model"] = LOAD_ERROR
        else:
            row, errors = parse_inputs(values)
            if not errors:
                try:
                    result = predict_row(row)
                except Exception as exc:
                    errors["_model"] = f"Prediksi gagal: {exc}"
    return render_template("index.html", features=FEATURES, groups=GROUPS, values=values, errors=errors,
                           result=result, info=model_info(), load_error=LOAD_ERROR, defaults=DEFAULTS)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if BUNDLE is None:
        return jsonify({"error": LOAD_ERROR}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Body harus berupa JSON objek berisi 21 fitur."}), 400
    row, errors = parse_inputs(payload)
    if errors:
        return jsonify({"error": "Input tidak valid.", "detail": errors}), 422
    try:
        return jsonify(predict_row(row))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({"model_loaded": BUNDLE is not None, "error": LOAD_ERROR})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
