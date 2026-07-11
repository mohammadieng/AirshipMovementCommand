import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

VIDEO_PATH = "footage/movebox.mp4"
scale_check = 16
leastbest_points = 30
frame_count = 1
coords_count = 0
window = 30
    
orb = cv2.ORB_create(
    nfeatures=1200,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    fastThreshold=7,
)

FLANN_INDEX_LSH = 6
index_params = dict(
    algorithm=FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1,
)
search_params = dict(checks=scale_check)
flann = cv2.FlannBasedMatcher(index_params, search_params)

kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], dtype=np.float32)
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.7
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.3
kf.errorCovPost = np.eye(4, dtype=np.float32)
kf.statePost = np.zeros((4, 1), dtype=np.float32)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, prev_frame = cap.read()
if not ret:
    raise RuntimeError("Video read error")

himg , wimg, zimg = prev_frame.shape
xCenter = int(wimg/2)
yCenter = int(himg/2)

def hav(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(h))


points = [(0,0)]

scale_m_per_pixel = 1

new_x, new_y = 0, 0

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp_prev, des_prev = orb.detectAndCompute(prev_gray, None)

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], marker='o', linestyle='-', color='b')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Realtime Movement Path")
ax.grid(True)

path = np.zeros((1, 2), dtype=np.float32)
x_min, x_max = -0.5, 0.5
y_min, y_max = -0.5, 0.5
traj_pix = []
xs = deque(maxlen=window)
ys = deque(maxlen=window)
    
H_global = np.eye(3, dtype=np.float32)
xtrack = 0
ytrack = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if len(xs) == window:
        smooth_x = sum(xs) / window
        smooth_y = sum(ys) / window
    else:
        smooth_x, smooth_y = new_x, new_y
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    pred = kf.predict()
    pred_x, pred_y = float(pred[0, 0]), float(pred[1, 0])
    
    measurement_ok = False
    if des is not None and des_prev is not None:
        matches = flann.knnMatch(des_prev, des, k=2)
        good = [m[0] for m in matches
                if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

        if len(good) >= leastbest_points:
            pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
            pts_curr = np.float32([kp[m.trainIdx].pt for m in good])
            H, _ = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 3.0)

            if H is not None:
                H_global = H @ H_global
                tx = H_global[0, 2] * scale_m_per_pixel
                ty = H_global[1, 2] * scale_m_per_pixel
                meas = np.array([[tx], [ty]], dtype=np.float32)
                kf.correct(meas)
                measurement_ok = True
    
    
    xs.append(new_x)
    ys.append(new_y)
    

    if measurement_ok:
        new_x, new_y = tx, ty
    else:
        new_x, new_y = pred_x, pred_y
    
    path = np.vstack((path, np.array([[smooth_x, smooth_y]], dtype=np.float32)))
    line.set_data(path[:, 0], path[:, 1])
    line.set_color('r')
    line.set_linewidth(2)

    line.set_data(path[:, 0], path[:, 1])

    if new_x < x_min: x_min = new_x
    if new_x > x_max: x_max = new_x
    if new_y < y_min: y_min = new_y
    if new_y > y_max: y_max = new_y
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.invert_xaxis()

    px = int(points[0][0] + new_x / scale_m_per_pixel)
    py = int(points[0][1] + new_y / scale_m_per_pixel)
    traj_pix.append((px, py))
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    
    frame = cv2.line(frame, (xCenter,0), (xCenter,himg), (0, 0, 255), 1)
    frame = cv2.line(frame, (0, yCenter), (wimg, yCenter), (0, 0, 255), 1)
    
    dx = smooth_x - xtrack
    dy = smooth_y - ytrack
    
    center_threshold = 20
    if smooth_x != 0 and smooth_y != 0:
        move_command = "STAY"
        
        if abs(dx) > center_threshold:
            if dx > 0: horizontal_direction = "RIGHT"
            else: horizontal_direction = "LEFT"
        else:
            horizontal_direction = "CENTER"
        
        if abs(dy) > center_threshold:
            if dy > 0: vertical_direction = "DOWN"
            else: vertical_direction = "UP"
        else:
            vertical_direction = "CENTER"
        
        if horizontal_direction != "CENTER" or vertical_direction != "CENTER":
            move_command = f"MOVE {horizontal_direction} {vertical_direction}"
        
        cv2.putText(frame, move_command, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
        
        cv2.imshow("Frame", frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == ord('q'):
            break
        elif k == ord('r'):
            xtrack = new_x
            ytrack = new_y
            print(xtrack, ytrack)
            path = np.zeros((1, 2), dtype=np.float32)

    kp_prev, des_prev = kp, des

cap.release()
cv2.destroyAllWindows()