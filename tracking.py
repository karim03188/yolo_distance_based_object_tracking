from collections import defaultdict
import random
import cv2
import pandas as pd
from ultralytics import YOLO
import math



# varable area of YOLO
pre_trined_model = YOLO('yolov8s.pt')
objects_name = open("coco.txt", "r")
obj_names_file = objects_name.read()
obj_names = obj_names_file.split("\n")


# varaible area of frame
frame_width = 1000
frame_height = 700


center_point_objs_last_frame = []
frame_count = 0

tracking_objects = {}
tracking_objects_history = defaultdict(list)
history_flage = False
history_expire = False

cureent_id_track_object = None

track_obj_id = 0
clicked_obj_id = None

click_empty_area_flag = False
click_empty_area_x = 0
click_empty_area_y = 0





def move_camera_text(x_obj, y_obj, frame_width, frame_height):
    print(f'x = {x_obj}\ny = {y_obj}')


# Tracking object area
def select_object_for_tracking(center_x, center_y, mouse_x, mouse_y):
    # Define the range or rectangle for clicking
    click_range = 30
    x1_range = center_x - click_range
    x2_range = center_x + click_range
    y1_range = center_y - click_range
    y2_range = center_y + click_range



    # Check if the click is within the range or rectangle
    if x1_range <= mouse_x <= x2_range and y1_range <= mouse_y <= y2_range:
        return True
    return False


# event handaling
def mouseClick(event, x, y, flags, param):
    global frame, cureent_id_track_object, tracking_objects, clicked_obj_id, click_empty_area_flag, click_empty_area_x, click_empty_area_y, history_flage, history_expire

    # loak camera to object
    if event == cv2.EVENT_LBUTTONDOWN:
        for object_id, pt in tracking_objects.items():
            center_x, center_y = pt
            if select_object_for_tracking(center_x, center_y, x, y):
                click_empty_area_flag = False
                click_empty_area_x = 0
                click_empty_area_y = 0
                clicked_obj_id = object_id
                cureent_id_track_object = object_id
                history_expire = False

    # relase camera loak to object
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked_obj_id = None
        click_empty_area_flag = True
        click_empty_area_x = x
        click_empty_area_y = y

    # show history line of object
    if event == cv2.EVENT_RBUTTONDOWN:
        if clicked_obj_id is not None:
            history_flage = True

    # hide history line of object
    if event == cv2.EVENT_RBUTTONDBLCLK:
        history_flage = False


# binding the event in window
cv2.namedWindow('window')
cv2.setMouseCallback('window', mouseClick)


# read a video from dirctory
cap = cv2.VideoCapture('vidyolov8.mp4')


while True:

    # if video end the while break
    ret, frame = cap.read()
    if not ret:
        break

    # count the frame and set center point to a list
    frame_count += 1
    center_points_cur_frame = []

    # get data from detected object in the frame
    frame = cv2.resize(frame, (frame_width, frame_height))
    results = pre_trined_model.predict(frame)
    
    data = results[0].boxes.data
    
    pandas_data = pd.DataFrame(data.cpu().numpy()).astype('float')

    # read all data from pandas DataFrame and draw rectangle put name and accurcy
    for index, row in pandas_data.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        name = obj_names[int(row[5])]

        # accurcy = math.ceil(row[4]*100)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # insert center point to a list for comper with prviws frame
        center_points_cur_frame.append((center_x, center_y))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        cv2.putText(frame, str(name), (x1+5, y1+12),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)




    # in first time last frame center point is empty
    if frame_count <= 1:
        for pt in center_points_cur_frame:
            tracking_objects[track_obj_id] = pt
            track_obj_id += 1




    # make a copy from center point of all detected object
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        # itterat in all object in tracking_objects dictionary for chaeck in current frame is exists or left
        for object_id, pt2 in tracking_objects_copy.items():

            # for chack object is exists or no
            object_exists = False

            # itterate in current frame center point of all object and culcualte the distace of points
            for pt in center_points_cur_frame_copy:

                # formula for achive distance bettwen to point current object from all object detected in all frames and no left
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # for check the distance bettwen tow object if less then 20 this tow object are same means no left the frame
                if distance < 20:

                    # save the new coordenat of object in the traking object dectionary
                    tracking_objects[object_id] = pt

                    # save the new coordenat of object in the traking history for draw a line
                    tracking_objects_history[object_id].append((pt[0], pt[1]))

                    # make the falge true means object still alife in the frame
                    object_exists = True

                    # remove the center point of object from the current frame for no checking again then continue
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue


            # if object left the frame meain the object expired remove from the tracking objects
            if not object_exists:
                tracking_objects.pop(object_id)
                if object_id in tracking_objects_history.keys():
                    tracking_objects_history.pop(object_id)
                    history_expire = True

                    if object_id == cureent_id_track_object:
                        cureent_id_track_object = None



        # same object detected in this frame and has not any id then save in racking objects dictionary and diclear new id
        for pt in center_points_cur_frame:
            track_obj_id += 1
            tracking_objects[track_obj_id] = pt




    # put id number as text in top of all object detectd
    for object_id, pt in tracking_objects.items():
        cv2.putText(frame, str(object_id), (pt[0]-5, pt[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)




    if cv2.waitKey(27) & 0xFF == 27:
        break




    else:
        if (clicked_obj_id is None) and (click_empty_area_flag == False):
            cv2.rectangle(frame, (320, 0), (1000, 50),
                          (255, 255, 255), thickness=cv2.FILLED)
            text = 'No Object selected for tracking'
            cv2.putText(frame, text, (400, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cx = frame_width//2
            cy = frame_height//2
            print('No object selected ')
            move_camera_text(cx, cy, frame_width, frame_height)
            print('\n')



        # when any object left the frame then remove all data belong the object from the history dictonary
        elif history_expire == True and cureent_id_track_object == None:
            cv2.rectangle(frame, (320, 0), (1000, 50),
                          (255, 255, 255), thickness=cv2.FILLED)
            # put the text in the top of scrren for help the user
            text = 'The object exit from camera area'
            cv2.putText(frame, text, (400, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cx = frame_width//2
            cy = frame_height//2

            # change direction of camera to center of screen
            print('Object left the frame camera foucse center ')
            move_camera_text(cx, cy, frame_width, frame_height)
            print('\n')

        # this cection of code for tracking the object loack the camera to object
        if (clicked_obj_id is not None) and (clicked_obj_id in tracking_objects.keys()):

            # get x point and y point of object
            cx, cy = tracking_objects[clicked_obj_id]

            # draw rectangle the top of screen for show same help message
            cv2.rectangle(frame, (320, 0), (1000, 50),(255, 255, 255), thickness=cv2.FILLED)

            # for show information message in the top ob window
            text = f'The camera track object number {clicked_obj_id}'
            b, g, r = random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255)
            cv2.putText(frame, text, (400, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, 0, r), 1)

            # for draw circle and lable the selected object
            cv2.circle(frame, (cx, cy), 20, (b, g, r), 2)
            cv2.circle(frame, (cx, cy), 8, (b, g, r), 1)
            cv2.line(frame, (cx - 14, cy), (cx + 14, cy), (b, g, r), 2)
            cv2.line(frame, (cx, cy - 14), (cx, cy + 14), (b, g, r), 2)

            # for change direction of camera
            print(f'Camera track object {clicked_obj_id}')
            move_camera_text(cx, cy, frame_width, frame_height)
            print('\n')


        if (clicked_obj_id is None) and (click_empty_area_flag == True):

            cv2.circle(frame, (click_empty_area_x,
                       click_empty_area_y), 5, (0, 255, 0), -1)

            cx = click_empty_area_x
            cy = click_empty_area_y

            print('Select empty area')
            move_camera_text(cx, cy, frame_width, frame_height)
            print('\n')

            cv2.rectangle(frame, (320, 0), (1000, 50),(255, 255, 255), thickness=cv2.FILLED)
            text = 'No Object selected for tracking'
            cv2.putText(frame, text, (400, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if clicked_obj_id is not None:
            if history_flage:
                rout = tracking_objects_history[clicked_obj_id]
                for i in range(len(rout)-1):
                    start_x, start_y = rout[i]
                    stop_x, stop_y = rout[i+1]
                    cv2.line(frame, (start_x, start_y),(stop_x, stop_y), (0, 0, 255), 3)


    center_point_objs_last_frame = center_points_cur_frame.copy()


    cv2.rectangle(frame, (0, 0), (345, 130),
                  (0, 0, 0), thickness=cv2.FILLED)

    text = 'Left click: Choice object for tracking'
    cv2.putText(frame, text, (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    text = 'Left double click: select any area point'
    cv2.putText(frame, text, (5, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    text = 'Esc key: exit'
    cv2.putText(frame, text, (5, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    text = 'Right Click: show history line'
    cv2.putText(frame, text, (5, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    text = 'Right double Click: hide history line'
    cv2.putText(frame, text, (5, 114),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    cv2.imshow("window", frame)


cap.release()
cv2.destroyAllWindows()




