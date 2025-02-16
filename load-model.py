import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Memuat model yang telah disimpan
model = load_model('./sampah_classifier_model-index5.h5')  # Ganti dengan path ke model Anda

# Daftar label kelas (sesuai urutan saat pelatihan model)
class_labels = ['cardboard', 'metal', 'paper', 'plastic']

# Fungsi untuk memprediksi gambar
def predict_image(image_path):
    # Membaca gambar dan mengubahnya menjadi format yang sesuai untuk model
    img = image.load_img(image_path, target_size=(150, 150))  # Menyesuaikan ukuran gambar dengan input model
    img_array = image.img_to_array(img)  # Mengonversi gambar menjadi array
    img_array = np.expand_dims(img_array, axis=0)  # Menambahkan dimensi batch untuk model

    # Melakukan prediksi
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Mendapatkan indeks kelas dengan probabilitas tertinggi
    predicted_class_label = class_labels[predicted_class_index]  # Mendapatkan label kelas dari indeks
    
    # Mengembalikan hasil prediksi (probabilitas dan label)
    return predicted_class_label, predictions[0]

# Contoh penggunaan
image_path = './test/test7.jpg'  # Ganti dengan path ke gambar yang ingin diprediksi
predicted_label, probabilities = predict_image(image_path)
print(f"Prediksi Kelas: {predicted_label}")
print(f"Probabilitas Tiap Kelas: {probabilities}")
