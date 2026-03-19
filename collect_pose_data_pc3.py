import cv2
import mediapipe as mp
import csv
import time

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils

# 開啟 CSV 檔案
file = open('pose_data_pc_multiclass.csv', 'w', newline='')
writer = csv.writer(file)
writer.writerow([f'x{i}' for i in range(33)] + [f'y{i}' for i in range(33)] + ['label'])

# 姿勢標籤對應字典
posture_labels = {
    0: "Correct post",
    1: "(head down)",
    2: "(turning)",
    3: "(lean)",
    4: "(head tilted)",
    5: "(sloping shoulder)",
    6: "(lying down)",
    7: "(hands up)",
}

def collect_samples(label):
    label_name = posture_labels.get(label, "未知姿勢")
    print(f"⏳ 3 秒後開始收集：{label} - {label_name}")
    time.sleep(3)
    print(f"🎬 開始收集 50 筆資料，每 0.2 秒一筆...")

    collected = 0
    total_samples = 50
    interval = 0.2  # 秒
    last_capture_time = time.time()
    start_time = time.time()

    while collected < total_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()
        if current_time - last_capture_time >= interval:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                x_coords = [lm.x for lm in results.pose_landmarks.landmark]
                y_coords = [lm.y for lm in results.pose_landmarks.landmark]
                writer.writerow(x_coords + y_coords + [label])
                collected += 1
                last_capture_time = current_time
                print(f"[✓] 已儲存第 {collected}/{total_samples} 筆：{label_name}")

        # 顯示畫面
        cv2.imshow("Posture Data Collection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    print(f"✅ 已完成 {total_samples} 筆：{label_name} 的資料收集！\n")

# 主迴圈
while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 顯示提示文字
    cv2.putText(image, "Press 0~5 for recording, Press 'q' to leave", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for key, name in posture_labels.items():
        cv2.putText(image, f"{key}: {name}", (10, 60 + 25 * key),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Posture Data Collection", image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord(str(i)) for i in posture_labels.keys()]:
        collect_samples(label=int(chr(key)))

# 結束處理
cap.release()
file.close()
cv2.destroyAllWindows()
print("✅ 資料收集完成！已儲存為 pose_data_pc_multiclass.csv")
