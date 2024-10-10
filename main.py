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

def mark_chemical_center(frame):
    """Allows the user to select the center of the chemical by clicking with the mouse."""
    chemical_center = None

    def mouse_click(event, x, y, flags, param):
        nonlocal chemical_center
        if event == cv2.EVENT_LBUTTONDOWN:
            chemical_center = (x, y)

    cv2.namedWindow('Mark Chemical Center')
    cv2.setMouseCallback('Mark Chemical Center', mouse_click)
    cv2.imshow('Mark Chemical Center', frame)

    while chemical_center is None:
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyWindow('Mark Chemical Center')
    return chemical_center

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# Capture the first frame and mark the chemical center
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
chemical_center = mark_chemical_center(frame)

# Lists to store time and distance data
object_data = {}
object_tracks = {}
next_id = 0

start_time = time.time()
warmup_duration = 1  # 5 seconds warm-up period

# Enable interactive mode for Matplotlib
plt.ion()
fig, ax = plt.subplots()
colors = plt.cm.rainbow(np.linspace(0, 1, 100))  # Maximum of 100 different colors

# For the heatmap
heatmap = np.zeros_like(frame[:, :, 0]).astype(float)

# Create folders for recording
record_folder = 'heatmap_records'
heatmap_overlay_folder = 'heatmap_overlay_records'
os.makedirs(record_folder, exist_ok=True)
os.makedirs(heatmap_overlay_folder, exist_ok=True)

# Variables for the timer
record_interval_with_detection = 3  # Recording interval in seconds when something is detected
last_record_time = time.time()

# Dictionary to track the last recorded times
last_recorded_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    fgmask = fgbg.apply(frame)

    # Filtering
    fgmask = cv2.medianBlur(fgmask, 5)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Skip detection during the warm-up period
    elapsed_time = time.time() - start_time
    if elapsed_time < warmup_duration:
        #cv2.putText(frame, "Warming up...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Heatmap', frame)
        if cv2.waitKey(30) == ord('q'):  # Press 'q' to exit
            break
        continue

    # Area Filtering
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for contour in contours:
        if cv2.contourArea(contour) < 150:
            continue  # Skip small areas

        # Find the center of the insects
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Calculate the distance (Euclidean distance)
            distance = np.sqrt((cX - chemical_center[0])**2 + (cY - chemical_center[1])**2)

            detected_objects.append((cX, cY, distance))

    # Object matching and identification
    for (cX, cY, distance) in detected_objects:
        object_id = next_id
        next_id += 1
        for oid, data in object_data.items():
            if np.sqrt((data[-1][1] - cX)**2 + (data[-1][2] - cY)**2) < 50:
                object_id = oid
                break

        if object_id not in object_data:
            object_data[object_id] = []
            object_tracks[object_id] = deque(maxlen=20)
            last_recorded_times[object_id] = None  # Initialize last recorded time as None for new insects

        current_time = datetime.datetime.now()
        if last_recorded_times[object_id] is None or (current_time - last_recorded_times[object_id]).total_seconds() >= 1:  # Check for recording every second
            current_time_str = current_time.strftime("%H:%M:%S")
            object_data[object_id].append((current_time_str, cX, cY, distance))
            object_tracks[object_id].append((cX, cY))
            last_recorded_times[object_id] = current_time  # Update the last recorded time

        # Draw rectangle and display distance
        (x, y, w, h) = cv2.boundingRect(np.array([[cX, cY]]))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{distance:.2f}", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {object_id}", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Mark the position of the chemical
    cv2.circle(frame, chemical_center, 5, (0, 0, 0), -1)

    # Update the heatmap
    for oid, track in object_tracks.items():
        for (x, y) in track:
            heatmap[y, x] += 1

    # Blur and normalize the heatmap
    heatmap_blurred = gaussian_filter(heatmap, sigma=10)
    heatmap_normalized = (heatmap_blurred - heatmap_blurred.min()) / (heatmap_blurred.max() - heatmap_blurred.min())

    # Colorize the heatmap (3 channels)
    heatmap_colored = (plt.cm.jet(heatmap_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Match dimensions
    heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

    # Mark the position of the chemical on the heatmap_colored (black)
    cv2.circle(heatmap_colored, chemical_center, 5, (0, 0, 0), -1)  # Black color (0, 0, 0)

    # Overlay the heatmap on the video (optional)
    heatmap_overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
    cv2.imshow('Heatmap', heatmap_overlay)

    # Check if any objects were detected
    if detected_objects:
        current_time = time.time()
        # If it's been at least 3 seconds since the last save, save the frame
        if current_time - last_record_time >= record_interval_with_detection:
            # Save both images at the same time
            overlay_filename = os.path.join(heatmap_overlay_folder, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_heatmap_overlay.png')
            colored_filename = os.path.join(record_folder, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
            cv2.imwrite(overlay_filename, heatmap_overlay)
            cv2.imwrite(colored_filename, heatmap_colored)
            last_record_time = current_time

    # Update the plot
    ax.clear()
    for oid, data in object_data.items():
        times = [datetime.datetime.strptime(d[0], "%H:%M:%S") for d in data]
        distances = [d[3] for d in data]
        ax.plot(times, distances, marker='o', linestyle='-', color=colors[oid % 100], label=f'ID {oid}')
    ax.set_title('Insect Distance to Chemical Center Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance (pixels)')
    ax.legend()
    ax.grid(True)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()
    plt.pause(0.001)


    if cv2.waitKey(30) == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Save the plot
save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plot_filename = f'insect_distance_plot_{save_time}.png'
plt.savefig(plot_filename)

# Disable interactive mode for Matplotlib
plt.ioff()
plt.show()

# Save the data to an Excel file
all_data = []
for oid, data in object_data.items():
    for entry in data:
        all_data.append([oid, entry[0], entry[3]])  # Only ID, time, and distance

df = pd.DataFrame(all_data, columns=['Object ID', 'Time', 'Distance'])
df.to_excel('insect_tracking_data.xlsx', index=False)
