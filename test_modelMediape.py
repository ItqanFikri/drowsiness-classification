import cv2
import numpy as np
import pandas as pd
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from ind_rnn import IndRNN
import mediapipe as mp
import time

# Fungsi untuk menghitung Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    eye = np.array(eye)  # Konversi ke numpy array
    A = np.linalg.norm(eye[1] - eye[5])  # Jarak vertikal antara titik 2 dan 6
    B = np.linalg.norm(eye[2] - eye[4])  # Jarak vertikal antara titik 3 dan 5
    C = np.linalg.norm(eye[0] - eye[3])  # Jarak horizontal antara titik 1 dan 4
    ear = (A + B) / (2.0 * C)
    return ear

# Fungsi untuk menghitung Mouth Aspect Ratio (MAR)
def calculate_mar(mouth):
    mouth = np.array(mouth)  # Konversi ke numpy array
    A = np.linalg.norm(mouth[1] - mouth[7])  # Jarak vertikal antara bibir atas dan bawah
    B = np.linalg.norm(mouth[2] - mouth[6])  # Jarak vertikal antara bibir atas dan bawah
    C = np.linalg.norm(mouth[3] - mouth[5])  # Jarak horizontal antara sudut bibir
    D = np.linalg.norm(mouth[0] - mouth[4])  # Jarak horizontal antara sudut bibir
    mar = (A + B + C) / (3.0 * D)
    return mar

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the pre-trained classification model (IndRNN atau model lainnya)
model = load_model('./Itqan/model_slide_window/IndRNNAug5400_Slide_revisi80.h5', custom_objects={'IndRNN': IndRNN}, compile=False)  # Pastikan model sudah dilatih dan disimpan

data = pd.read_csv('./Itqan/data_dummy_ngantuk.csv')
data = data[['ear_x', 'mar_x']].iloc[:5400].to_numpy()

# Load data CSV untuk inisialisasi EAR dan MAR

# Pisahkan EAR dan MAR ke dalam list
ear_sequence = data[:, 0].tolist()  # EAR
mar_sequence = data[:, 1].tolist()  # MAR
# print(ear_sequence, mar_sequence)

# Panjang urutan yang dibutuhkan oleh model
sequence_length = 5400

# Cek prediksi awal menggunakan data CSV
print("Melakukan prediksi awal dari data CSV...")
input_sequence = np.array([ear_sequence, mar_sequence]).T
input_sequence = input_sequence.reshape(1, 5400, 2)

print(input_sequence.shape)

print("Dimensi input sebelum prediksi:", input_sequence.shape)

print("Prediksi awal:", input_sequence)

prediction = model.predict(input_sequence)
print(f'Initial prediction from CSV data: {prediction}')

pred_label = np.argmax(prediction)
print(f'Predicted class: {pred_label}')  # Print kelas yang diprediksi


# # initial_prediction = model.predict(x_test)
# print(f'Initial prediction from CSV data: {initial_prediction}')

# Start time untuk menghitung FPS
start_time = time.time()
frame_count = 0
fps = 0  # Inisialisasi FPS dengan nilai awal 0

# Mengakses kamera
cap = cv2.VideoCapture('F:/Dataset TA/Video/01/10.mov')

# cap = cv2.VideoCapture(1)


# Specify the desired display window size

# Target lebar baru
window_width = 420

# Ambil ukuran video asli
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Hitung faktor skala
scale_factor = window_width / original_width
window_height = int(original_height * scale_factor)

# Tambahkan nama file untuk menyimpan output
output_file = "F:/Recorder/Listing/output.txt"

# Buat file baru atau hapus isinya jika sudah ada
with open(output_file, "w") as file:
    file.write("Frame,EAR,MAR,Predicted Class,FPS\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1  # Menambah jumlah frame yang diproses
        
    # Ubah gambar menjadi RGB untuk MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark mata kiri
            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
            left_eye = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in left_eye_landmarks]

            # Landmark mata kanan
            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
            right_eye = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in right_eye_landmarks]

            # Landmark mulut
            mouth_landmarks = [face_landmarks.landmark[i] for i in [78, 81, 13, 311, 308, 402, 14, 178]]
            mouth = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in mouth_landmarks]

            # Menghitung EAR dan MAR
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear = (ear_left + ear_right) / 2.0
            ear = ear + 0.028780171747515366
        

            # Menghitung MAR
            mar = calculate_mar(mouth)
            
            # preprocessing data EAR dan MAR
            earNew = ear
            marNew = mar
                        
            # Pembatasan EAR dan MAR
            if earNew > 0.35:
                earNew = 0.35
            elif earNew < 0.2 and earNew != 0:
                earNew = 0.2

            if marNew > 0.9:
                marNew = 0.9
            elif marNew < 0.15 and marNew != 0:
                marNew = 0.15
    
            if earNew > 0:
                earNew = earNew / 0.35
            if marNew > 0:
                marNew = marNew / 0.9

            # if len(ear_sequence) < sequence_length:
            ear_sequence.pop(0)
            ear_sequence.append(earNew)

            # Print the first five elements or sublists
            print(ear_sequence[:5])
            print(ear_sequence[-5:])
            print(len(ear_sequence))

            # if len(mar_sequence) < sequence_length:
            mar_sequence.pop(0)
            mar_sequence.append(marNew)

            # Print the first five elements or sublists
            print(mar_sequence[:5])
            print(mar_sequence[-5:])
            print(len(mar_sequence))
            
            # Bentuk input sekuens yang sesuai dengan model
            input_sequence = np.array([ear_sequence, mar_sequence]).T
            # input_sequence = np.transpose(input_sequence, (1, 2, 0))
            # Reshape the data to the desired shape
            input_sequence = input_sequence.reshape(1, 5400, 2)
            
            print(frame_count)
            
            if frame_count % 1800 == 0:
                # Check input_sequence shape and some of its values
                print(f"Input sequence shape for prediction: {input_sequence.shape}")
                print(f"Last 5 elements of EAR in input sequence: {input_sequence[:, -5:, 0]}")
                print(f"Last 5 elements of MAR in input sequence: {input_sequence[:, -5:, 1]}")
            
                # Perform prediction
                # prediction = model.predict(np.expand_dims(input_sequence, axis=0))  # Shape (1, 5400, 2)
                
                # Perform prediction
                # prediction = model.predict(np.expand_dims(input_sequence, axis=0))  # Shape (1, 5400, 2)
                prediction = model.predict(input_sequence)
                pred_label = np.argmax(prediction)
                print(f"Predicted class: {pred_label}")    
                
            # Menampilkan hasil EAR, MAR, dan prediksi pada frame
            cv2.putText(frame, f'EAR: {ear:.2f}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f'MAR: {mar:.2f}', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Gambar mata dan mulut pada frame
            cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            
            # # Menampilkan hasil prediksi
            # pred_label = np.argmax(prediction)
            # print(f'Predicted class: {pred_label}')  # Print kelas yang diprediksi
            
            cv2.putText(frame, f'Prediction: {pred_label}', (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, f'Prediction: {prediction}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # drowsiness_status = "Drowsy" if prediction[0] > 0.1 else "Alert"
            # cv2.putText(frame, f'Status: {drowsiness_status}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # drowsiness_status = "Drowsy" if prediction[0] > 0.1 else "Alert"
            # cv2.putText(frame, f'Status: {drowsiness_status}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Simpan hasil ke file
            with open(output_file, "a") as file:
                file.write(f"{frame_count}///{ear:.2f}///{mar:.2f}///{pred_label}///{fps:.2f}\n")
            
        
    # Hitung FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    
    
    # Tampilkan FPS di frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    frame = cv2.resize(frame, (window_width, window_height))
    # Tampilkan frame
    cv2.imshow('Frame', frame)

    # Break loop ketika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release dan tutup jendela
cap.release()
cv2.destroyAllWindows()