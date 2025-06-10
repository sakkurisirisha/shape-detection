import cv2
import numpy as np
import math

drawing = False
points = []
canvas = np.ones((550, 500, 3), dtype=np.uint8) * 255  # Updated canvas size
temp_canvas = canvas.copy()

def draw_shape(event, x, y, flags, param):
    global drawing, points, temp_canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
        temp_canvas = canvas.copy()
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points.append((x, y))
        temp_canvas = canvas.copy()
        if len(points) > 1:
            cv2.polylines(temp_canvas, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))
        cv2.polylines(canvas, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        temp_canvas = canvas.copy()

def detect_shape(image, points):
    if len(points) < 3:
        cv2.putText(image, "Draw a full shape before detection.", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

    contour = np.array(points).reshape((-1, 1, 2)).astype(np.int32)
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
    x, y, w, h = cv2.boundingRect(approx)
    num_vertices = len(approx)
    
    shape = "Unidentified"

    # Check if it's a circle using circularity check
    # First, compute the contour area and perimeter (arc length)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter != 0:  # To avoid division by zero
        circularity = 4 * math.pi * area / (perimeter ** 2)
        if circularity > 0.8:  # Threshold for circle-like shapes
            shape = "Circle"
    
    if shape == "Unidentified":
        if num_vertices == 3:
            shape = "Triangle"
        elif num_vertices == 4:
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = approx[:, 0]
            side1 = math.hypot(x2 - x1, y2 - y1)
            side2 = math.hypot(x3 - x2, y3 - y2)
            side3 = math.hypot(x4 - x3, y4 - y3)
            side4 = math.hypot(x1 - x4, y1 - y4)
            sides = [side1, side2, side3, side4]
            avg_side = sum(sides) / 4

            aspect_ratio = w / float(h)
            if all(abs(s - avg_side) < 0.1 * avg_side for s in sides):
                shape = "Square"
            elif 0.95 <= aspect_ratio <= 1.05:
                shape = "Diamond"
            else:
                shape = "Rectangle"
        elif num_vertices == 5:
            shape = "Pentagon"
        elif num_vertices == 6:
            shape = "Hexagon"
        elif num_vertices == 8:
            shape = "Octagon"

    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    cv2.putText(image, f"Detected: {shape}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image

# Setup window and instructions
cv2.putText(canvas, "Draw a shape with your mouse", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
cv2.putText(canvas, "Press 'd' to detect, 'c' to clear, 'q' to quit", (10, 480),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas", draw_shape)

while True:
    cv2.imshow("Canvas", temp_canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d') and points:
        canvas = detect_shape(canvas, points)
        temp_canvas = canvas.copy()
        points = []
    elif key == ord('c'):
        canvas = np.ones((550, 500, 3), dtype=np.uint8) * 255  # Updated canvas size
        cv2.putText(canvas, "Draw a shape with your mouse", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.putText(canvas, "Press 'd' to detect, 'c' to clear, 'q' to quit", (10, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        temp_canvas = canvas.copy()
        points = []
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
