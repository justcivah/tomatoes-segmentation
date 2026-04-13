import cv2
import torch
import numpy as np
import model
import os
import time

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ────────────────────────────────────────────────────────────────
model = model.YNet().to(device)

checkpoint_path = os.path.join(
    'models/2026-03-31_09-53-54/',
    model.__class__.__name__,
    'best_model.pt'
)

checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in checkpoint['model_state_dict'].items()
}
model.load_state_dict(state_dict)
model.eval()

# ── Webcam ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_time = time.time()

# ── Inference loop ────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    # Preprocess
    frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 360))
    img = (torch.from_numpy(frame_resized).float() / 255.0)
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(img)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.85).float()

    # Postprocess
    pred_np = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

    # Overlay FPS on webcam feed
    display = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show
    cv2.imshow("Webcam", display)
    cv2.imshow("Prediction", pred_np)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()