
# simple controller with onboard camera + keyboard + PS4 controller

from controller import Display, Keyboard
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import pygame

# configuration constants
DEBOUNCE_TIME = 0.1
MAX_ANGLE = 0.5
MAX_SPEED = 250
SPEED_INCR = 5
ANGLE_INCR = 0.05

# PS4 controller tuning
DEADZONE = 0.10
CONTROLLER_MAX_SPEED = 80      # safer than 250 for joystick control
REVERSE_MAX_SPEED = -20

# PS4 button map, typical Windows / pygame mapping
BTN_X = 0
BTN_CIRCLE = 1
BTN_TRIANGLE = 2
BTN_SQUARE = 3
BTN_L1 = 4
BTN_R1 = 5
BTN_SHARE = 8
BTN_OPTIONS = 9
BTN_PS = 10


def apply_deadzone(value, deadzone=DEADZONE):
    if abs(value) < deadzone:
        return 0.0
    return value


def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image


def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img


def display_image(display, image):
    image_rgb = np.dstack((image, image, image))
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


def init_ps_controller():
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()

    if joystick_count == 0:
        print("No PS controller detected. Keyboard only mode.")
        return None

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Controller detected: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}, Buttons: {joystick.get_numbuttons()}")

    return joystick


def main():
    speed = 10
    angle = 0.0
    last_press = {}

    autonomous_mode = False

    robot = Car()
    driver = Driver()

    timestep = int(robot.getBasicTimeStep())

    camera = robot.getDevice("camera")
    camera.enable(timestep)

    display_img = Display("display_image")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    joystick = init_ps_controller()

    while robot.step() != -1:
        image = get_image(camera)
        gray_image = greyscale_cv2(image)
        # resize to match Webots Display size
        gray_small = cv2.resize(gray_image, (640, 320))
        display_image(display_img, gray_small)

        current_time = time.time()

        # ----------------------------------------------------
        # Keyboard control
        # ----------------------------------------------------
        key = keyboard.getKey()

        if key != -1:
            if key in last_press and (current_time - last_press[key] < DEBOUNCE_TIME):
                pass
            else:
                last_press[key] = current_time

                if key == keyboard.UP:
                    if speed < MAX_SPEED:
                        speed += SPEED_INCR
                        print("keyboard up")

                elif key == keyboard.DOWN:
                    if speed >= SPEED_INCR:
                        speed -= SPEED_INCR
                        print("keyboard down")

                elif key == keyboard.RIGHT:
                    angle += ANGLE_INCR
                    angle = min(angle, MAX_ANGLE)
                    print("keyboard right")

                elif key == keyboard.LEFT:
                    angle -= ANGLE_INCR
                    angle = max(angle, -MAX_ANGLE)
                    print("keyboard left")

                elif key == ord('A'):
                    current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                    file_name = current_datetime + ".png"
                    print("Image taken")
                    camera.saveImage(os.getcwd() + "/" + file_name, 1)

                elif key == ord('S'):
                    autonomous_mode = not autonomous_mode
                    print(f"Autonomous mode: {autonomous_mode}")

        # ----------------------------------------------------
        # PS4 controller control
        # ----------------------------------------------------
        if joystick is not None:
            pygame.event.pump()

            # Typical PS4 mapping:
            # Axis 0 = left stick horizontal
            # Axis 1 = left stick vertical
            # Axis 2 = L2 trigger
            # Axis 3 = right stick horizontal
            # Axis 4 = right stick vertical
            # Axis 5 = R2 trigger

            left_stick_x = apply_deadzone(joystick.get_axis(0))

            # Steering with left stick
            angle = left_stick_x * MAX_ANGLE

            # Triggers: usually range from -1 to +1
            l2 = joystick.get_axis(2)
            r2 = joystick.get_axis(5)

            throttle = (r2 + 1) / 2
            brake = (l2 + 1) / 2

            speed = throttle * CONTROLLER_MAX_SPEED - brake * abs(REVERSE_MAX_SPEED)

            # X button toggles autonomous mode
            if joystick.get_button(BTN_X):
                if "ps_x" not in last_press or current_time - last_press["ps_x"] > 0.5:
                    autonomous_mode = not autonomous_mode
                    last_press["ps_x"] = current_time
                    print(f"Autonomous mode: {autonomous_mode}")

            # Square button saves camera image
            if joystick.get_button(BTN_SQUARE):
                if "ps_square" not in last_press or current_time - last_press["ps_square"] > 0.5:
                    current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                    file_name = current_datetime + ".png"
                    print("Image taken from PS controller")
                    camera.saveImage(os.getcwd() + "/" + file_name, 1)
                    last_press["ps_square"] = current_time

        # ----------------------------------------------------
        # Future self-driving section
        # ----------------------------------------------------
        if autonomous_mode:
            # Later you can replace this with lane detection / PID output
            # Example:
            # angle = pid_angle
            # speed = autonomous_speed
            pass

        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()


    ## in order to create the connection to the Webot ,, need to run this in the terminal shell.
    #(venv) PS D:\ML\Projects\Project_5_MR4010.10_Navegacion\MR4010.10_Navegacion> & "C:\Program Files\Webots\msys64\mingw64\bin\webots-controller.exe" .\src\simple_controller_act_2_1_V1.1_Bluetooth.py --stdout-redirect
    #to enable the keyboar in webot you need to click the webot window

    #Left stick left/right = steering
    #R2 = throttle
    #L2 = brake / reverse
    #X = toggle autonomous mode
    #Square = take image
    #Keyboard still works in parallel