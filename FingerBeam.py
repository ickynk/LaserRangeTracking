# -*- coding: utf-8 -*-
# FingerBeam: convert any beamer projection into an interactive screen.
# Updated 2025 — Python 3, OpenCV 4+, PyAutoGUI

import cv2
import numpy as np
import pyautogui

# globals
kernel          = np.ones((5, 5), np.uint8)
subproc         = 'Set image corners'
mousedown       = False
mouse_x, mouse_y = 0, 0
color_to_detect = None
corners         = []

# --- Mouse / Corner callbacks ----------------------------------------------

def click_corner(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONUP:
        corners.append([x, y])
        print(corners)

def safe_destroy(win_name):
    """Destroy window only if it exists and is visible."""
    try:
        # returns >=0 if window exists; -1 if not
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(win_name)
    except cv2.error:
        pass


def pick_color(event, x, y, flags, frame):
    global mouse_x, mouse_y, color_to_detect
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONUP:
        hsv_px = cv2.cvtColor(np.uint8([[frame[y, x]]]), cv2.COLOR_BGR2HSV)[0][0]
        color_to_detect = hsv_px
        print('Selected HSV:', color_to_detect)

# --- Thresholding & Tracking -----------------------------------------------

def getThresImage(frame):
    """Blur → HSV → red‐laser threshold → dilate."""
    # 1) smooth a bit to reduce noise
    blur = cv2.GaussianBlur(frame, (3, 3), 0)
    # 2) convert to HSV
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 3) red falls in two hue ranges: 0–10 and 170–180
    lower1 = np.array([  0, 150, 150], dtype=np.uint8)
    upper1 = np.array([ 10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 150, 150], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # 4) clean up and return
    return cv2.dilate(mask, kernel, iterations=2)



def track_color(frame):
    """Find largest contour → move / click mouse in projected space."""
    global mousedown

    thresh = getThresImage(frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if subproc == 'Mouse control' and mousedown:
            pyautogui.mouseUp()
            mousedown = False
            print('mouseup')
        return thresh

    # pick the largest blob
    best = max(contours, key=cv2.contourArea)
    M = cv2.moments(best)
    if M['m00'] == 0:
        return thresh

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # draw feedback
    cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 5)
    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 4)

    if subproc == 'Mouse control':
        scX, scY = ps.screenXY(cx, cy)
        pyautogui.moveTo(scX, scY)
        if not mousedown:
            pyautogui.mouseDown()
            mousedown = True
            print('mousedown')

    return thresh

# --- Perspective mapping ----------------------------------------------

def order_points(pts):
    """Sort points: tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class PS:
    """
    A class defining the projection transformation necessary to map the image from the beamer
    into the screen so that the mouse pointer can be moved adequately.
    """
    def __init__(self, corners):
        self.set_corners(corners)
        w, h = pyautogui.size()
        self.screen_corners = np.array([
            [0,    0],
            [w-1,  0],
            [w-1, h-1],
            [0,   h-1],
        ], dtype=np.float32)
        self.screen_size = (w, h)

    def set_corners(self, corners):
        if len(corners) >= 4:
            self.corners = np.float32(corners[-4:])

    def screenXY(self, x, y):
        M = cv2.getPerspectiveTransform(order_points(self.corners), self.screen_corners)
        pt = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), M
        )[0][0]
        scX = int(np.clip(pt[0], 0, self.screen_size[0]-1))
        scY = int(np.clip(pt[1], 0, self.screen_size[1]-1))
        return scX, scY

# --- Main ------------------------------------------------------------------

# create video capture
capture = cv2.VideoCapture(0, cv2.CAP_MSMF)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
capture.set(cv2.CAP_PROP_FPS, 60)

# load corners from file if available
# load corners from file
corners = []
try:
    with open("corners.dat", 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue
            xs, ys = parts
            try:
                x = int(float(xs))
                y = int(float(ys))
            except ValueError:
                continue
            corners.append([x, y])
except FileNotFoundError:
    pass

ps = PS(corners)

# windows setup
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if   key in (ord('q'), 27):          # q or Esc
        break
    elif key == ord('c'):
        subproc = 'Set image corners'
    elif key == ord('p'):
        subproc = 'Pick color to track'
    elif key == ord('t'):
        subproc = 'Mouse test'
    elif key == ord('m'):
        if len(ps.corners) >= 4:
            subproc = 'Mouse control'
        else:
            print('Define projection corners first.')
            subproc = 'Set image corners'

    if subproc == 'Set image corners':
        for cx, cy in corners[-4:]:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        cv2.setMouseCallback('frame', click_corner, frame)

    elif subproc == 'Pick color to track':
        cv2.setMouseCallback('frame', pick_color, frame)
        try:
            zoom = frame[mouse_y-10:mouse_y+11, mouse_x-10:mouse_x+11]
            zoom = cv2.resize(zoom, (210, 210), interpolation=cv2.INTER_AREA)
            cv2.rectangle(zoom, (95, 95), (105, 105), tuple(255 - frame[mouse_y, mouse_x]), 1)
            cv2.imshow('zoom', zoom)
        except Exception:
            pass

    elif subproc in ('Mouse test', 'Mouse control'):
        thresh = track_color(frame)
        if subproc == 'Mouse test':
            cv2.imshow('thresh', thresh)

    # housekeeping windows
    if subproc != 'Pick color to track':
        safe_destroy('zoom')
    if subproc not in ('Mouse test', 'Mouse control'):
        safe_destroy('thresh')
        cv2.putText(
            frame, subproc, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow('frame', frame)
    else:
        safe_destroy('frame')

# save corners
with open("corners.dat", 'w') as f:
    for x, y in ps.corners:
        f.write(f"{x},{y}\n")

# clean up
capture.release()
cv2.destroyAllWindows()
# -*- coding: utf-8 -*-
# FingerBeam: convert any beamer projection into an interactive screen.
# Updated 2025 — Python 3, OpenCV 4+, PyAutoGUI

import cv2
import numpy as np
import pyautogui

# globals
kernel          = np.ones((5, 5), np.uint8)
subproc         = 'Set image corners'
mousedown       = False
mouse_x, mouse_y = 0, 0
color_to_detect = None
corners         = []

# --- Mouse / Corner callbacks ----------------------------------------------

def click_corner(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONUP:
        corners.append([x, y])
        print(corners)


def pick_color(event, x, y, flags, frame):
    global mouse_x, mouse_y, color_to_detect
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONUP:
        hsv_px = cv2.cvtColor(np.uint8([[frame[y, x]]]), cv2.COLOR_BGR2HSV)[0][0]
        color_to_detect = hsv_px
        print('Selected HSV:', color_to_detect)

# --- Thresholding & Tracking -----------------------------------------------

def getThresImage(frame):
    """Blur → HSV → color threshold → dilate."""
    blur = cv2.blur(frame, (5, 5))
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    if color_to_detect is not None:
        h, s, v = color_to_detect
        lower = np.array((max(h * 0.85, 0), 50, 50), dtype=np.uint8)
        upper = np.array((min(h * 1.15, 179), 255, 255), dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
    else:
        # fallback: detect bright green
        mask = cv2.inRange(hsv, np.array((50, 32, 64)), np.array((90, 255, 255)))
    return cv2.dilate(mask, kernel, iterations=1)


def track_color(frame):
    """Find largest contour → move / click mouse in projected space."""
    global mousedown

    thresh = getThresImage(frame)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        if subproc == 'Mouse control' and mousedown:
            pyautogui.mouseUp()
            mousedown = False
            print('mouseup')
        return thresh

    # pick the largest blob
    best = max(contours, key=cv2.contourArea)
    M = cv2.moments(best)
    if M['m00'] == 0:
        return thresh

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # draw feedback
    cv2.circle(frame, (cx, cy), 8, (0, 0, 0), 5)
    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), 4)

    if subproc == 'Mouse control':
        scX, scY = ps.screenXY(cx, cy)
        pyautogui.moveTo(scX, scY)
        if not mousedown:
            pyautogui.mouseDown()
            mousedown = True
            print('mousedown')

    return thresh

# --- Perspective mapping ----------------------------------------------

def order_points(pts):
    """Sort points: tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class PS:
    """
    A class defining the projection transformation necessary to map the image from the beamer
    into the screen so that the mouse pointer can be moved adequately.
    """
    def __init__(self, corners):
        self.set_corners(corners)
        w, h = pyautogui.size()
        self.screen_corners = np.array([
            [0,    0],
            [w-1,  0],
            [w-1, h-1],
            [0,   h-1],
        ], dtype=np.float32)
        self.screen_size = (w, h)

    def set_corners(self, corners):
        if len(corners) >= 4:
            self.corners = np.float32(corners[-4:])

    def screenXY(self, x, y):
        M = cv2.getPerspectiveTransform(order_points(self.corners), self.screen_corners)
        pt = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), M
        )[0][0]
        scX = int(np.clip(pt[0], 0, self.screen_size[0]-1))
        scY = int(np.clip(pt[1], 0, self.screen_size[1]-1))
        return scX, scY

# --- Main ------------------------------------------------------------------

# create video capture
capture = cv2.VideoCapture(0, cv2.CAP_MSMF)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
capture.set(cv2.CAP_PROP_FPS, 60)

# load corners from file if available
try:
    with open("corners.dat", 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue
            xs, ys = parts
            try:
                x = int(float(xs))
                y = int(float(ys))
            except ValueError:
                continue
            corners.append([x, y])
except FileNotFoundError:
    pass

ps = PS(corners)

# windows setup
cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if   key in (ord('q'), 27):          # q or Esc
        break
    elif key == ord('c'):
        subproc = 'Set image corners'
    elif key == ord('p'):
        subproc = 'Pick color to track'
    elif key == ord('t'):
        subproc = 'Mouse test'
    elif key == ord('m'):
        if len(ps.corners) >= 4:
            subproc = 'Mouse control'
        else:
            print('Define projection corners first.')
            subproc = 'Set image corners'

    if subproc == 'Set image corners':
        for cx, cy in ps.corners[-4:]:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
        cv2.setMouseCallback('frame', click_corner, frame)

    elif subproc == 'Pick color to track':
        cv2.setMouseCallback('frame', pick_color, frame)
        try:
            zoom = frame[mouse_y-10:mouse_y+11, mouse_x-10:mouse_x+11]
            zoom = cv2.resize(zoom, (210, 210), interpolation=cv2.INTER_AREA)
            cv2.rectangle(zoom, (95, 95), (105, 105), tuple(255 - frame[mouse_y, mouse_x]), 1)
            cv2.imshow('zoom', zoom)
        except Exception:
            pass

    elif subproc in ('Mouse test', 'Mouse control'):
        thresh = track_color(frame)
        if subproc == 'Mouse test':
            cv2.imshow('thresh', thresh)

        # housekeeping windows
    if subproc != 'Pick color to track':
        safe_destroy('zoom')

    if subproc not in ('Mouse test', 'Mouse control'):
        safe_destroy('thresh')

        # draw and show the main frame
        cv2.putText(
            frame, subproc, (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow('frame', frame)
    else:
        # hide main frame in test/control modes
        safe_destroy('frame')

# save corners
with open("corners.dat", 'w') as f:
    for x, y in ps.corners:
        f.write(f"{x},{y}\n")

# clean up
capture.release()
cv2.destroyAllWindows()
