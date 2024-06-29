import cv2
import numpy as np

# Görüntü dosyasının yolu
image_path = "bocekler3.jpg"

# Görüntüyü oku
image = cv2.imread(image_path)

# Görüntüyü HSV (Hue, Saturation, Value) renk uzayına dönüştür
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Böceklerin renk aralığını belirle (kahverengi tonları için bir örnek)
lower_color = np.array([5, 50, 50])
upper_color = np.array([15, 255, 255])

# Renk aralığında maskeyi oluştur
mask = cv2.inRange(hsv, lower_color, upper_color)

# Maskeyi kullanarak görüntüyü filtrele
result = cv2.bitwise_and(image, image, mask=mask)

# Kontur bulma
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Konturları çiz
for cnt in contours:
    if cv2.contourArea(cnt) < 500:  # Küçük konturları filtreleme
        continue

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Kırmızı dikdörtgen

# Sonucu göster ve kaydet
cv2.imshow('Bugs Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

