import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import pandas as pd

def kimyasal_merkezini_isaretle(frame):
    """Kullanıcının kimyasalın merkezini fare tıklamasıyla seçmesini sağlar."""
    kimyasal_merkez = None

    def fare_tiklama(event, x, y, flags, param):
        nonlocal kimyasal_merkez
        if event == cv2.EVENT_LBUTTONDOWN:
            kimyasal_merkez = (x, y)

    cv2.namedWindow('Kimyasal Merkezini İşaretle')
    cv2.setMouseCallback('Kimyasal Merkezini İşaretle', fare_tiklama)
    cv2.imshow('Kimyasal Merkezini İşaretle', frame)

    while kimyasal_merkez is None:
        if cv2.waitKey(1) == ord('q'):  # 'q' tuşuna basarak çıkış
            break

    cv2.destroyWindow('Kimyasal Merkezini İşaretle')
    return kimyasal_merkez

cap = cv2.VideoCapture('bugs1.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# İlk kareyi al ve kimyasal merkezini işaretle
ret, frame = cap.read()
kimyasal_merkez = kimyasal_merkezini_isaretle(frame)

# Zaman ve uzaklık verilerini saklamak için listeler
object_data = {}
object_tracks = {}
next_id = 0

start_time = time.time()

# Matplotlib'in interaktif modunu etkinleştir
plt.ion()
fig, ax = plt.subplots()
colors = plt.cm.rainbow(np.linspace(0, 1, 100))  # Maksimum 100 farklı renk

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    # Filtreleme (Daha önceki filtreleme adımlarını da ekleyebilirsiniz)
    fgmask = cv2.medianBlur(fgmask, 5)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Alan Filtreleme
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue  # Küçük alanları atla

        # Böceklerin merkezini bul
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Uzaklığı hesapla (Öklid uzaklığı)
            uzaklik = np.sqrt((cX - kimyasal_merkez[0])**2 + (cY - kimyasal_merkez[1])**2)

            detected_objects.append((cX, cY, uzaklik))

    # Nesne eşleme ve kimliklendirme
    for (cX, cY, uzaklik) in detected_objects:
        object_id = next_id
        next_id += 1
        for oid, data in object_data.items():
            if np.sqrt((data[-1][1] - cX)**2 + (data[-1][2] - cY)**2) < 50:
                object_id = oid
                break

        if object_id not in object_data:
            object_data[object_id] = []
            object_tracks[object_id] = deque(maxlen=20)

        current_time = time.time() - start_time
        object_data[object_id].append((current_time, cX, cY, uzaklik))
        object_tracks[object_id].append((cX, cY))

        # Dikdörtgen içine alma ve uzaklığı yazdırma
        (x, y, w, h) = cv2.boundingRect(np.array([[cX, cY]]))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{uzaklik:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {object_id}", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # İz yollarını çizme
        for i in range(1, len(object_tracks[object_id])):
            if object_tracks[object_id][i - 1] is None or object_tracks[object_id][i] is None:
                continue
            cv2.line(frame, object_tracks[object_id][i - 1], object_tracks[object_id][i], colors[object_id % 100], 2)

    # Kimyasalın yerini işaretle
    cv2.circle(frame, kimyasal_merkez, 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgmask)

    # Grafiği güncelle
    ax.clear()
    for oid, data in object_data.items():
        times = [d[0] for d in data]
        distances = [d[3] for d in data]
        ax.plot(times, distances, marker='o', linestyle='-', color=colors[oid % 100], label=f'ID {oid}')
    ax.set_title('Böceklerin Kimyasal Merkezine Uzaklığına Göre Zaman Grafiği')
    ax.set_xlabel('Zaman (saniye)')
    ax.set_ylabel('Uzaklık (piksel)')
    ax.legend()
    ax.grid(True)
    plt.pause(0.001)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Matplotlib'in interaktif modunu kapat
plt.ioff()
plt.show()

# Verileri Excel dosyasına kaydet
all_data = []
for oid, data in object_data.items():
    for entry in data:
        all_data.append([oid, entry[0], entry[1], entry[2], entry[3]])

df = pd.DataFrame(all_data, columns=['Object ID', 'Time (s)', 'X', 'Y', 'Distance'])
df.to_excel('bocek_izleme_verileri.xlsx', index=False)
