import cv2
from mtcnn import MTCNN

# MTCNN 모델 초기화
detector = MTCNN()

# 카메라 영상 처리
cap = cv2.VideoCapture(4)  # 인덱스는 필요에 따라 0, 1 등으로 변경 가능

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 얼굴 감지
    faces = detector.detect_faces(frame)
    
    # 감지된 얼굴에 사각형 표시
    for face in faces:
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height
        # 사각형 그리기
        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
        
        # 예측된 얼굴 위치를 표시
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 실시간 화면 표시
    cv2.imshow("Live Face Detection", frame)

    # 종료 조건 (ESC 키)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
