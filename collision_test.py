import cv2
import torch
import time
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ─── REAL-TIME WEBCAM PARAMETERS ───────────────────────────────────────────────
DETECT_INTERVAL          = 3         # do heavy detect+depth every 3 frames
FRAME_WIDTH, FRAME_HEIGHT= 640, 480
TRACKER_CFG              = 'bytetrack.yaml'
HISTORY_LENGTH           = 5

# ─── WARNING THRESHOLDS ─────────────────────────────────────────────────────────
CLASS_WHITELIST          = {'car','truck','bus','person'}
SCALE_FACTOR             = 15.0      # depth_inv→meters
TTC_THRESHOLD            = 3.0       # seconds
DISTANCE_THRESHOLD       = 2.0       # meters (immediate distance warning)
MIN_SPEED                = 0.2       # m/s
MIN_CONF                 = 0.5       # confidence
MIN_AREA                 = 1500      # px²
WARNING_PERSIST_FRAMES   = 8
MAX_DISTANCE_CAP         = np.inf
VELOCITY_CHANGE_TH       = 2.0       # m/s sudden Δv
# ────────────────────────────────────────────────────────────────────────────────

# LED debounce state (stub)
warning_counter = 0
led_state       = False

def led_on():
    print("LED ON")   # GPIO.HIGH

def led_off():
    print("LED OFF")  # GPIO.LOW

# ─── MODEL SETUP ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MiDaS_small for speed, half precision
midas      = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
transforms = torch.hub.load("intel-isl/MiDaS","transforms")
transform  = transforms.small_transform
midas      = midas.to(device).eval().half()

# YOLOv8n detection-only, half precision
model = YOLO('yolov8n.pt')
model.model.half()
# ────────────────────────────────────────────────────────────────────────────────

# history buffers
dist_history = {}  # tid → deque[(timestamp, distance)]
vel_history  = {}  # tid → deque[(timestamp, v_rel)]

# open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

frame_count = 0
last_results= None
last_depth  = None
t0          = time.time()
fps_text    = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # ─── heavy inference every N frames ────────────────────────────────────────
    if frame_count % DETECT_INTERVAL == 0:
        # 1) Detect + track
        t1 = time.time()
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            results = model.track(
                source=frame,
                tracker=TRACKER_CFG,
                persist=True,
                device=device,
                verbose=False
            )[0]
        t2 = time.time()

        # 2) Depth at half-res → upsample
        small = cv2.resize(frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        inp   = transform(rgb).to(device).half()
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=small.shape[:2],
                mode='bilinear',
                align_corners=False
            ).squeeze()
        dm        = pred.cpu().numpy()
        norm      = (dm - dm.min())/(dm.max() - dm.min())
        depth_inv = cv2.resize(
            (1.0 - norm).astype(np.float32),
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )
        t3 = time.time()

        print(f"[PERF] detect:{t2-t1:.3f}s depth:{t3-t2:.3f}s", flush=True)

        last_results = results
        last_depth   = depth_inv
    else:
        results   = last_results
        depth_inv = last_depth

    warnings = []  # (tid, ttc_or_None, (x1,y1,x2,y2), reason)
    now      = time.time()

    # ─── process each track ──────────────────────────────────────────────────────
    if results is not None:
        for box in results.boxes:
            if box.id is None:
                continue
            tid = int(box.id.cpu().item())

            # unpack & filter by confidence/size
            x1,y1,x2,y2 = map(int, box.xyxy.cpu().numpy().flatten())
            conf        = float(box.conf.cpu().item())
            if conf < MIN_CONF or (x2-x1)*(y2-y1) < MIN_AREA:
                continue

            # class filter
            cls_idx  = int(box.cls.cpu().item())
            cls_name = results.names[cls_idx]
            if cls_name not in CLASS_WHITELIST:
                continue

            # draw class+ID
            cv2.putText(frame, f"{cls_name} ID:{tid}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 1, cv2.LINE_AA)

            # sample bottom-center for distance
            xc  = (x1+x2)//2
            yb  = min(FRAME_HEIGHT-1, y2-1)
            win = depth_inv[yb-2:yb+3, xc-2:xc+3].flatten()
            if win.size == 0:
                continue
            dist_m = float(np.median(win) * SCALE_FACTOR)
            if dist_m > MAX_DISTANCE_CAP:
                continue

            # immediate distance warning
            if dist_m < DISTANCE_THRESHOLD:
                warnings.append((tid, None, (x1,y1,x2,y2), 'TOO_CLOSE'))

            # update distance history
            dh = dist_history.setdefault(tid, deque(maxlen=HISTORY_LENGTH))
            dh.append((now, dist_m))

            # compute relative speed & TTC
            if len(dh) >= 2:
                (t0_,d0),(t1_,d1) = dh[-2], dh[-1]
                dt = t1_ - t0_
                if dt > 0:
                    v_rel = (d0 - d1)/dt

                    # overlays
                    cv2.putText(frame, f"D:{d1:.2f}m", (x1, y2+20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,0), 1)
                    cv2.putText(frame, f"v:{v_rel:.2f}m/s", (x1, y2+40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255,255,0), 1)

                    if v_rel > MIN_SPEED:
                        ttc = d1/v_rel
                        cv2.putText(frame, f"TTC:{ttc:.1f}s", (x1, y2+60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,255,255), 1)

                        # update velocity history
                        vh = vel_history.setdefault(tid, deque(maxlen=HISTORY_LENGTH))
                        vh.append((now, v_rel))
                        delta_v = (vh[-1][1]-vh[-2][1]) if len(vh)>=2 else 0.0

                        if ttc < TTC_THRESHOLD and delta_v > VELOCITY_CHANGE_TH:
                            warnings.append((tid, ttc, (x1,y1,x2,y2), 'COLLISION'))

    # ─── debounce & LED ─────────────────────────────────────────────────────────
    if warnings:
        warning_counter += 1
    else:
        warning_counter = 0

    if warning_counter >= WARNING_PERSIST_FRAMES and not led_state:
        led_on();  led_state = True
    elif warning_counter == 0 and led_state:
        led_off(); led_state = False

    # ─── draw warnings after stable debounce ────────────────────────────────────
    if warning_counter >= WARNING_PERSIST_FRAMES:
        for _, _, (x1,y1,x2,y2), reason in warnings:
            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,0,255), 2)
            text = "TOO CLOSE" if reason=='TOO_CLOSE' else "COLLISION WARNING"
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 2)

    # ─── PiP depth map ───────────────────────────────────────────────────────────
    if depth_inv is not None:
        depth_vis = (plt.cm.plasma(depth_inv/depth_inv.max())[:,:,:3]*255).astype(np.uint8)
        fh, fw    = frame.shape[:2]
        nh        = fh//3
        nw        = int(nh * depth_vis.shape[1]/depth_vis.shape[0])
        pip       = cv2.resize(depth_vis, (nw, nh))
        frame[fh-nh-20:fh-20, fw-nw-20:fw-20] = pip

    # ─── FPS display ─────────────────────────────────────────────────────────────
    if frame_count % 10 == 0:
        t4 = time.time()
        fps_text = f"FPS: {10/(t4-t0):.1f}"
        t0 = t4
    cv2.putText(frame, fps_text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,255), 1)

    cv2.imshow("ADAS Real-Time", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
