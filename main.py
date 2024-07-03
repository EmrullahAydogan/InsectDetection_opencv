import cv2
import numpy as np
from matplotlib import pyplot as plt

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

# Resmi yükle
image_path = 'bugs7.jpg'
image = cv2.imread(image_path)

# Ön işleme
preprocessed_image = preprocess_image(image)

# Segmentasyon
segmented_image = segment_image(preprocessed_image)

# Konturları bul
contours = find_contours(segmented_image)

# Konturları çiz
output_image = draw_contours(image.copy(), contours)

# Sonuçları görselleştir
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Cockroaches')
plt.show()
