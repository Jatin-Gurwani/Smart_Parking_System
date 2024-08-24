# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
from Testing_Scripts import DetectionProcess as DP

check_licence_flag = False
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    while True:

        Camera1_obj = cv2.VideoCapture(0)
        Camera1_obj.set(3, 1280)
        Camera1_obj.set(4, 720)
        success, Camera1 = Camera1_obj.read()
        Camera1_cv2, check_licence_flag = DP.vehicledetection(Camera1)
        cv2.imshow("SmartPark Tracker", Camera1)
        cv2.waitKey(1)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
