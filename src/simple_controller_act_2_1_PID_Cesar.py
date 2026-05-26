#simple controller with onboard camera

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time

#configuration constants
DEBOUNCE_TIME = 0.1 #100 milliseconds
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing example
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

#Detect yellow lane line using HSV color filter
def detect_yellow_line(image):
    # Convert from BGRA to BGR
    bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Convert from BGR to HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Yellow color range
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([35, 255, 255])

    # Create mask for yellow color
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    return yellow_mask

#Display image on onboard display
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


#Detect edges using Canny
def canny_cv2(gray_image):
    edges = cv2.Canny(gray_image, 50, 150)
    return edges

#Define region of interest using fillPoly
def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)

    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2, height // 2)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    roi_image = cv2.bitwise_and(image, mask)
    return roi_image

#Detect straight lines using HoughLinesP
def hough_lines(image):
    lines = cv2.HoughLinesP(
        image,
        1,
        np.pi / 180,
        20,
        minLineLength=10,
        maxLineGap=30
    )
    return lines

#Simple PID controller
def pid_controller(error, previous_error, integral):
    KP = 0.01
    KI = 0.0001
    KD = 0.001

    integral = integral + error
    derivative = error - previous_error

    output = (KP * error) + (KI * integral) + (KD * derivative)

    previous_error = error

    return output, previous_error, integral

#Calculate smallest error from detected lines
def calculate_error(lines, setpoint):
    if lines is None:
        return 0

    min_error = float("inf")

    for line in lines:
        x1, y1, x2, y2 = line[0]

        #Ignore mostly horizontal lines
        if abs(y2 - y1) < 10:
            continue

        line_midpoint = (x1 + x2) // 2
        error = line_midpoint - setpoint

        if abs(error) < abs(min_error):
            min_error = error

    if min_error == float("inf"):
        return 0

    return min_error



# main
def main():
    speed = 50
    angle = 0.0
    last_press = {}

    #PID variables
    previous_error = 0
    integral = 0

    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Convert image to greyscale
        grey_image = greyscale_cv2(image)

        # Detect yellow line first
        yellow_image = detect_yellow_line(image)

        # Detect edges only from yellow mask
        edges_image = canny_cv2(yellow_image)

        # Apply region of interest
        roi_image = region_of_interest(yellow_image)

        # Detect lines using Hough Transform
        lines = hough_lines(roi_image)

        # Display processed image
        display_image(display_img, roi_image)

        # Setpoint: middle of the camera image
        setpoint = grey_image.shape[1] // 2

        # Calculate smallest error from detected lines
        error = calculate_error(lines, setpoint)

        # PID calculates steering angle
        # If no valid line is detected, drive straight
        if lines is None or error == 0:
            angle = 0.0
        else:
            angle, previous_error, integral = pid_controller(
                error,
                previous_error,
                integral
            )

        # Limit steering angle
        if angle > MAX_ANGLE:
            angle = MAX_ANGLE
        elif angle < -MAX_ANGLE:
            angle = -MAX_ANGLE

        print("Lines:", 0 if lines is None else len(lines), 
            "Error:", error, 
            "Angle:", angle)

        #to reduce rebounds
        current_time = time.time()

        # Read keyboard
        key = keyboard.getKey()
        #print("Key:", key)

        if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
            continue # Ignore rebound

        #pressed key accepted, update
        last_press[key] = current_time

        # Keep only A key to save images
        if key == ord('A') or key == ord('a'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage("D:\\WebotsImagenes\\" + file_name, 100)

        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()