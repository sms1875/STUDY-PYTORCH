import cv2
import numpy as np
import tensorflow as tf

# 미리 학습된 모델 로드
model = tf.keras.models.load_model('mnist_model.h5')

# 전역 변수
rect_start = None  # 사각형 시작 좌표
rect_end = None    # 사각형 끝 좌표
drawing = False    # 드래그 상태 추적

# 마우스 콜백 함수 (사각형 지정)
def draw_rectangle(event, x, y, flags, param):
    global rect_start, rect_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end = (x, y)
            img_copy = img.copy()
            cv2.rectangle(img_copy, rect_start, rect_end, (0, 255, 0), 2)
            cv2.imshow("Live Camera", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect_end = (x, y)
        cv2.rectangle(img, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow("Live Camera", img)

# 카메라 영상 처리
cap = cv2.VideoCapture(4)

cv2.namedWindow("Live Camera")
cv2.setMouseCallback("Live Camera", draw_rectangle)
while True:
    ret, img = cap.read()
    if not ret:
        break

    # 사각형 영역이 지정되었으면 해당 부분에 테두리를 그림
    if rect_start and rect_end:
        # 사각형 테두리 표시
        cv2.rectangle(img, rect_start, rect_end, (0, 255, 0), 2)
        
        # 해당 부분을 잘라내서 28x28로 크기 조정 후 모델에 입력
        roi = img[rect_start[1]:rect_end[1], rect_start[0]:rect_end[0]]
        if roi.size != 0:
            # 이미지를 28x28로 크기 변경
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi_resized_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            roi_resized_gray = roi_resized_gray / 255.0  # 정규화
            roi_resized_gray = np.expand_dims(roi_resized_gray, axis=-1)
            roi_resized_gray = np.expand_dims(roi_resized_gray, axis=0)

            # 모델 예측
            predictions = model.predict(roi_resized_gray)
            predicted_class = np.argmax(predictions)

            # 예측된 숫자 출력
            cv2.putText(img, f'Predicted: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 실시간 화면 표시
    cv2.imshow("Live Camera", img)

    # 종료 조건 (ESC 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()