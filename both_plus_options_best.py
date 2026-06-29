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

# Buffers
GENDER_BUFFER = deque(maxlen=6)
AGE_BUFFER = deque(maxlen=15)

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

print("🎥 Camera started!")
print("Controls:")
print("   E = Ensemble (Both models)")
print("   1 = EfficientNet-B3 only")
print("   2 = ResNet-50 only")
print("   Q = Quit")

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

    # FPS Calculation
    curr_frame_time = time.time()
    fps = int(1 / (curr_frame_time - prev_frame_time))
    prev_frame_time = curr_frame_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
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
            else:  # resnet
                gender_pred, age_pred = model_res(input_tensor)
                model_name = "ResNet-50"

            gender_probs = torch.softmax(gender_pred, dim=1)
            gender_conf, _ = torch.max(gender_probs, 1)
            gender_conf = gender_conf.item()

            current_gender = "Male" if torch.argmax(gender_pred, 1).item() == 0 else "Female"
            current_age = int(round(age_pred.item()))

        # Buffering
        if gender_conf > 0.60:
            GENDER_BUFFER.append(current_gender)
            AGE_BUFFER.append(current_age)

        # Stable predictions
        stable_gender = max(set(GENDER_BUFFER), key=GENDER_BUFFER.count) if GENDER_BUFFER else current_gender

        if len(AGE_BUFFER) >= 5:
            stable_age = int(round(np.mean(AGE_BUFFER)))
            stable_age = max(10, min(90, stable_age))
            age_std = np.std(list(AGE_BUFFER)) if len(AGE_BUFFER) > 1 else 0
            age_conf = max(0, 1.0 - (age_std / 10.0))
        else:
            stable_age = current_age
            age_conf = 0.5

        color = (0, 255, 0) if stable_gender == "Male" else (255, 20, 147)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{stable_gender}, {stable_age}y", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

        cv2.putText(frame, f"{model_name} | Gender: {gender_conf:.0%}", 
                    (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"Age Conf: {age_conf:.0%}", 
                    (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)

    # Top Info
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