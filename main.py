import cv2
import numpy as np

# Video dosyasının yolu veya kamera numarası (0, 1, ...) olarak belirtin
video_path = "bocekvideo.mp4"  # 0 ise birinci kamerayı kullanır, farklı bir yol verilebilir.

# Video yakalayıcıyı başlat
cap = cv2.VideoCapture(video_path)

# Renk aralığı için değerler
lower_color = np.array([5, 50, 50])
upper_color = np.array([15, 255, 255])

while True:
    # Bir kareyi yakala
    ret, frame = cap.read()
    if not ret:
        break

    # Kareyi HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Belirtilen renk aralığında maskeyi oluştur
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Maskeyi kullanarak görüntüyü filtrele
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Konturları bulma
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Konturları çizme
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:  # Küçük konturları filtreleme
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Kırmızı dikdörtgen

    # Sonucu göster
    cv2.imshow('Bugs Detection', frame)

    # Çıkış için 'q' tuşuna basıldığını kontrol et
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizleme
cap.release()
cv2.destroyAllWindows()
