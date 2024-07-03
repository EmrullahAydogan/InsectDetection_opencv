import cv2
import numpy as np

def preprocess_image(image):
    # Gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Gaussian bulanıklaştırma uygulayın
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

def segment_image(image):
    # Adaptif eşikleme uygulayın
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # Morfolojik işlemler: Açma (erozyon + genişleme)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Büyük alanları kapatmak için genişletme
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    return sure_bg

def find_contours(image):
    # Konturları bulun
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contours(original_image, contours):
    # Konturları çiz
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Çok küçük konturları filtrele
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return original_image

# Video dosyasını veya kamera beslemesini aç
video_path = 'bugs1.mp4'
cap = cv2.VideoCapture(video_path)

# Video yazıcı ayarları
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # Ön işleme
    preprocessed_image = preprocess_image(frame)

    # Segmentasyon
    segmented_image = segment_image(preprocessed_image)

    # Konturları bul
    contours = find_contours(segmented_image)

    # Konturları çiz
    output_frame = draw_contours(frame.copy(), contours)

    # İşlenmiş kareyi kaydet
    out.write(output_frame)

    # Sonuçları göstermek için kareyi göster
    cv2.imshow('Detected Cockroaches', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
out.release()
cv2.destroyAllWindows()
