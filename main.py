import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
from collections import deque
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
import os

def kimyasal_merkezini_isaretle(frame):
    """Kullanıcının kimyasalın merkezini fare tıklamasıyla seçmesini sağlar."""
    kimyasal_merkez = None

    def fare_tiklama(event, x, y, flags, param):
        nonlocal kimyasal_merkez
        if event == cv2.EVENT_LBUTTONDOWN:
            kimyasal_merkez = (x, y)

    cv2.namedWindow('Kimyasal Merkezini Isaretle')
    cv2.setMouseCallback('Kimyasal Merkezini Isaretle', fare_tiklama)
    cv2.imshow('Kimyasal Merkezini Isaretle', frame)

    while kimyasal_merkez is None:
        if cv2.waitKey(1) == ord('q'):  # 'q' tuşuna basarak çıkış
            break

    cv2.destroyWindow('Kimyasal Merkezini Isaretle')
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

# Yoğunluk haritası için
heatmap = np.zeros_like(frame[:, :, 0]).astype(float)

# Kayıt klasörü oluşturma
kayit_klasoru = 'yogunluk_haritasi_kayitlari'
os.makedirs(kayit_klasoru, exist_ok=True)

# Zamanlayıcı için değişkenler
kayit_araligi = 10  # Saniye cinsinden kayıt aralığı
son_kayit_zamani = time.time()

# Son kayıt zamanlarını takip etmek için bir sözlük
last_recorded_times = {}

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
            last_recorded_times[object_id] = None  # Yeni böcek için son kayıt zamanı None olarak başlat

        current_time = datetime.datetime.now()
        if last_recorded_times[object_id] is None or (current_time - last_recorded_times[object_id]).total_seconds() >= 1:  # Saniyede bir kayıt kontrolü
            current_time_str = current_time.strftime("%H:%M:%S")
            object_data[object_id].append((current_time_str, cX, cY, uzaklik))
            object_tracks[object_id].append((cX, cY))
            last_recorded_times[object_id] = current_time  # Son kayıt zamanını güncelle

        # Dikdörtgen içine alma ve uzaklığı yazdırma
        (x, y, w, h) = cv2.boundingRect(np.array([[cX, cY]]))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{uzaklik:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {object_id}", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Kimyasalın yerini işaretle
    cv2.circle(frame, kimyasal_merkez, 5, (0, 0, 255), -1)

    #cv2.imshow('Frame', frame)
    #cv2.imshow('FG Mask', fgmask)

    # Yoğunluk haritasını kaydetme
    simdiki_zaman = time.time()
    if simdiki_zaman - son_kayit_zamani >= kayit_araligi:
        dosya_adi = os.path.join(kayit_klasoru, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
        cv2.imwrite(dosya_adi, heatmap_colored)
        son_kayit_zamani = simdiki_zaman


    # Yoğunluk haritasını güncelle
    for oid, track in object_tracks.items():
        for (x, y) in track:
            heatmap[y, x] += 1

    # Yoğunluk haritasını bulanıklaştır ve normalize et
    heatmap_blurred = gaussian_filter(heatmap, sigma=10)
    heatmap_normalized = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min())

    # Yoğunluk haritasını renklendir (3 kanallı)
    heatmap_colored = (plt.cm.jet(heatmap_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Boyutları eşitle
    heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

    # Kimyasalın yerini heatmap_colored üzerinde işaretle (siyah)
    cv2.circle(heatmap_colored, kimyasal_merkez, 5, (0, 0, 0), -1)  # Siyah renk (0, 0, 0)

    # Yoğunluk haritasını videoya ekle (isteğe bağlı)
    heatmap_overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
    cv2.imshow('Yogunluk Haritasi', heatmap_overlay)



    # Grafiği güncelle
    ax.clear()
    for oid, data in object_data.items():
        times = [datetime.datetime.strptime(d[0], "%H:%M:%S") for d in data]
        distances = [d[3] for d in data]
        ax.plot(times, distances, marker='o', linestyle='-', color=colors[oid % 100], label=f'ID {oid}')
    ax.set_title('Böceklerin Kimyasal Merkezine Uzaklığına Göre Zaman Grafiği')
    ax.set_xlabel('Zaman (H:M:S)')
    ax.set_ylabel('Uzaklık (piksel)')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.pause(0.001)

    # Yoğunluk haritasını ayrı pencerede göster
    #cv2.imshow('Yoğunluk Haritası', heatmap_colored)

    if cv2.waitKey(30) == ord('q'):  # 'q' tuşuna basarak çıkış
        break

cap.release()
cv2.destroyAllWindows()

# Matplotlib'in interaktif modunu kapat
plt.ioff()
plt.show()

# Yoğunluk haritasını kaydet (isteğe bağlı)
cv2.imwrite('yogunluk_haritasi.png', heatmap_overlay)
cv2.imwrite('yogunluk_haritasi2.png', heatmap_colored)

# Verileri Excel dosyasına kaydet
all_data = []
for oid, data in object_data.items():
    for entry in data:
        all_data.append([oid, entry[0], entry[1], entry[2], entry[3]])

df = pd.DataFrame(all_data, columns=['Object ID', 'Time (H:M:S)', 'X', 'Y', 'Distance'])
df.to_excel('bocek_izleme_verileri.xlsx', index=False)