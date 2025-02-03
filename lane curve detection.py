import cv2
import numpy as np
import winsound

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

vidcap = cv2.VideoCapture("challenge.mp4")

while True:
    success, frame = vidcap.read()
    if not success:
        break
    
    frame = cv2.resize(frame, (640, 480))

    # Perspective points
    tl = (222, 387)
    bl = (70, 472)
    tr = (400, 380)
    br = (538, 472)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [640, 0], [640, 480]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))

    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    histogram = np.sum(mask[mask.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    y = 472
    lx = []
    rx = []

    msk = mask.copy()

    while y > 0:
        left_img = mask[y-40:y, left_base-50:left_base+50]
        contours, _ = cv2.findContours(left_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                lx.append(left_base - 50 + cx)
                left_base = left_base - 50 + cx

        right_img = mask[y-40:y, right_base-50:right_base+50]
        contours, _ = cv2.findContours(right_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                rx.append(right_base - 50 + cx)
                right_base = right_base - 50 + cx

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        y -= 40

    if len(lx) > 1 and len(rx) > 1:
        left_curve = np.polyfit(np.arange(len(lx)), lx, 2)[0]
        right_curve = np.polyfit(np.arange(len(rx)), rx, 2)[0]
        curve_threshold = 0.001  # Adjust this threshold as needed

        print(f"Left Curve: {left_curve}, Right Curve: {right_curve}")  # Debugging output

        if abs(left_curve) > curve_threshold or abs(right_curve) > curve_threshold:
            print("Curvature exceeded threshold, beeping...")  # Debugging output
            winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

    cv2.imshow("Original", frame)
    cv2.imshow("Bird's Eye View", transformed_frame)
    cv2.imshow("Lane Detection - Image Thresholding", mask)
    cv2.imshow("Lane Detection - Sliding Windows", msk)

    if cv2.waitKey(10) == 27:
        break

vidcap.release()
cv2.destroyAllWindows()

