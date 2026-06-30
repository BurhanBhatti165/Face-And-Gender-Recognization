import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import sys
from collections import deque
import numpy as np
import time

print("🚀 Starting Age & Gender Real-time System with Model Switching...")

device = torch.device('cpu')

# ===================== MODEL DEFINITIONS =====================
class AgeGenderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.gender_head(x), self.age_head(x)


class AgeGenderModel_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(0.5)
        self.gender_head = nn.Linear(in_features, 2)
        self.age_head = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)
        return self.gender_head(x), self.age_head(x)


# ===================== LOAD MODELS =====================
model_eff = AgeGenderModel().to(device)
model_res = AgeGenderModel_ResNet().to(device)

try:
    model_eff.load_state_dict(torch.load('EfficientNet-B3.pth', map_location=device))
    model_res.load_state_dict(torch.load('ResNet-50.pth', map_location=device))
    model_eff.eval()
    model_res.eval()
    print("✅ Both models loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", e)
    sys.exit()

# ===================== TRANSFORM =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ===================== TRACK STATE PER FACE =====================
# Instead of one global buffer, we keep a small state object per tracked face
# so a new face never inherits another person's history.
class FaceTrack:
    def __init__(self):
        self.gender_buffer = deque(maxlen=12)   # longer window = fewer flips
        self.age_buffer = deque(maxlen=20)
        self.locked_gender = None                # hysteresis lock
        self.locked_gender_streak = 0
        self.last_seen = time.time()
        self.smoothed_box = None                  # exponentially smoothed bbox

    def update_box(self, box, alpha=0.4):
        if self.smoothed_box is None:
            self.smoothed_box = np.array(box, dtype=float)
        else:
            self.smoothed_box = alpha * np.array(box, dtype=float) + (1 - alpha) * self.smoothed_box
        return tuple(self.smoothed_box.astype(int))


tracks = {}  
next_track_id = 0
MAX_TRACK_AGE = 1.0  

GENDER_LOCK_FRAMES = 8       
GENDER_CONF_THRESHOLD = 0.65  

# Current Mode
mode = "ensemble"
model_name = "Ensemble"

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Could not open webcam!")
    sys.exit()

frame_skip = 2
frame_count = 0
prev_frame_time = time.time()


def iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0


def match_track(box, tracks, threshold=0.3):
    best_id, best_iou = None, threshold
    for tid, (trk, last_box) in tracks.items():
        score = iou(box, last_box)
        if score > best_iou:
            best_id, best_iou = tid, score
    return best_id


print("🎥 Camera started!")
print("Controls:")
print("   E = Ensemble (Both models)")
print("   1 = EfficientNet-B3 only")
print("   2 = ResNet-50 only")
print("   Q = Quit")

# tracks now stores tid -> (FaceTrack, last_raw_box)
tracks = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        cv2.imshow('Age & Gender Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    curr_frame_time = time.time()
    fps = int(1 / max(1e-6, curr_frame_time - prev_frame_time))
    prev_frame_time = curr_frame_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))

    seen_ids = set()

    for (x, y, w, h) in faces:
        raw_box = (x, y, w, h)
        tid = match_track(raw_box, tracks)

        if tid is None:
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = (FaceTrack(), raw_box)

        track, _ = tracks[tid]
        track.last_seen = curr_frame_time
        tracks[tid] = (track, raw_box)
        seen_ids.add(tid)

        # Smooth the bounding box itself — this alone kills a lot of visible "jitter"
        sx, sy, sw, sh = track.update_box(raw_box)
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = max(1, sw), max(1, sh)

        face = frame[sy:sy + sh, sx:sx + sw]
        if face.size == 0:
            continue
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        input_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            if mode == "ensemble":
                g_eff, a_eff = model_eff(input_tensor)
                g_res, a_res = model_res(input_tensor)
                gender_pred = (g_eff + g_res) / 2
                age_pred = (a_eff + a_res) / 2
                model_name = "Ensemble"
            elif mode == "eff":
                gender_pred, age_pred = model_eff(input_tensor)
                model_name = "EfficientNet-B3"
            else:
                gender_pred, age_pred = model_res(input_tensor)
                model_name = "ResNet-50"

            gender_probs = torch.softmax(gender_pred, dim=1)
            gender_conf, _ = torch.max(gender_probs, 1)
            gender_conf = gender_conf.item()

            current_gender = "Male" if torch.argmax(gender_pred, 1).item() == 0 else "Female"
            current_age = int(round(age_pred.item()))

        # Only feed confident frames into the buffer
        if gender_conf > GENDER_CONF_THRESHOLD:
            track.gender_buffer.append(current_gender)
        track.age_buffer.append(current_age)  # age buffer doesn't need a gate, std-filter handles it

        # ---- Gender: majority vote + hysteresis lock ----
        if track.gender_buffer:
            majority = max(set(track.gender_buffer), key=track.gender_buffer.count)
        else:
            majority = current_gender

        if track.locked_gender is None:
            track.locked_gender = majority
            track.locked_gender_streak = 0
        elif majority == track.locked_gender:
            track.locked_gender_streak = 0
        else:
            track.locked_gender_streak += 1
            if track.locked_gender_streak >= GENDER_LOCK_FRAMES:
                track.locked_gender = majority
                track.locked_gender_streak = 0

        stable_gender = track.locked_gender

        # ---- Age: rolling mean with outlier rejection ----
        if len(track.age_buffer) >= 5:
            ages = np.array(track.age_buffer)
            mean, std = np.mean(ages), np.std(ages)
            filtered = ages[np.abs(ages - mean) <= max(std, 1e-6) * 2] if std > 0 else ages
            stable_age = int(round(np.mean(filtered))) if len(filtered) else int(round(mean))
            stable_age = max(10, min(90, stable_age))
            age_conf = max(0, 1.0 - (std / 10.0))
        else:
            stable_age = current_age
            age_conf = 0.5

        color = (0, 255, 0) if stable_gender == "Male" else (255, 20, 147)

        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color, 2)
        cv2.putText(frame, f"{stable_gender}, {stable_age}y", (sx, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, f"{model_name} | Gender: {gender_conf:.0%}",
                    (sx, sy + sh + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Age Conf: {age_conf:.0%}",
                    (sx, sy + sh + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)

    # Drop stale tracks (face left frame)
    tracks = {tid: t for tid, t in tracks.items()
              if (tid in seen_ids) or (curr_frame_time - t[0].last_seen < MAX_TRACK_AGE)}

    cv2.putText(frame, f"Mode: {model_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow('Age & Gender Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        mode = "ensemble"
    elif key == ord('1'):
        mode = "eff"
    elif key == ord('2'):
        mode = "resnet"

cap.release()
cv2.destroyAllWindows()