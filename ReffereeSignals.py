import cv2
import mediapipe as mp
import math
import os
from tkinter import *
from datetime import datetime


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=2)


def detectPose(image, pose):

    # Create a copy of the input image.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Initialize a list to store the detected landmarks.
    landmarks = []

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)

        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    # Return the output image and the found landmarks.
    return output_image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

def calculateAngleYZ(landmark1, landmark2, landmark3):

    # Get the required landmarks coordinates.
    _, y1, z1 = landmark1
    _, y2, z2 = landmark2
    _, y3, z3 = landmark3


    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, z3 - z2) - math.atan2(y1 - y2, z1 - z2))

    # Check if the angle is less than zero.
    if angle < 0:
        angle += 360

    return angle

def calculatePartAnglesXY(landmarks):
    # shoulder-rame
    # elbow-lakat
    # wrist-rucni zglob
    # racunanje ugla pozicije leve ruke u odnosu na: rame-lakat-zglob
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # racunanje ugla pozicije desne ruke u odnosu na: rame-lakat-zglob
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # hip-kuk
    # racunanje pozicije levog ramena u odnosu na lakat-rame-kuk(ugao leve ruke u odnosu na telo)
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         # problem je sto kuk i rame nisu u liniji, a potreban mi je taj ugao da bude 0 kako bih racuna pravi ugao ruke u odnosu na zemlju, ne u odnosu na kuk
                                         (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][0],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][2]))

    # racunanje pozicije desnog ramena u odnosu na lakat-rame-kuk(ugao desne ruke u odnosu na telo)
    right_shoulder_angle = calculateAngle((landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][2]),
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # knee-koleno
    # ankle-nozni zglob
    # racunanje pozicije levog kolena u odnosu na: kuk-koleno-zglob
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # racunanje pozicije desnog kolena u odnosu na: kuk-koleno-zglob
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    return left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle

def calculatePartAnglesYZ(landmarks):
    # hip-kuk
    # racunanje pozicije levog ramena u odnosu na lakat-rame-kuk(ugao leve ruke u odnosu na telo)
    left_shoulder_angleYZ = calculateAngleYZ(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][0],
                                              landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1],
                                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][2]))

    # racunanje pozicije desnog ramena u odnosu na lakat-rame-kuk(ugao desne ruke u odnosu na telo)
    right_shoulder_angleYZ = calculateAngleYZ((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][2]),
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    # racunanje ugla pozicije desne ruke u odnosu na: rame-lakat-zglob
    right_elbow_angleYZ = calculateAngleYZ(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    return left_shoulder_angleYZ, right_shoulder_angleYZ, right_elbow_angleYZ
def classifyPose(landmarks, output_image):

    label = 'Nepoznata poza'

    color = (0, 0, 255)

    # ----------------------------------------------------------------------------------------------------------------
    left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle, left_knee_angle, right_knee_angle = calculatePartAnglesXY(landmarks)
    left_shoulder_angleYZ, right_shoulder_angleYZ, right_elbow_angleYZ = calculatePartAnglesYZ(landmarks)
    # ----------------------------------------------------------------------------------------------------------------

    #provera da li su obe noge ispravljene
    if ((left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195) and
            (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][2]>landmarks[mp_pose.PoseLandmark.NOSE.value][2] and landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][2]<-landmarks[mp_pose.PoseLandmark.NOSE.value][2]) and
            (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][2]>landmarks[mp_pose.PoseLandmark.NOSE.value][2] and landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][2]<-landmarks[mp_pose.PoseLandmark.NOSE.value][2])):
        # provera da li su leva i desna ruka ispravljene/razlika kada je ruka napred i u stranu je xy i yz
        if left_elbow_angle > 160 and left_elbow_angle < 210 and ((right_elbow_angle > 160 and right_elbow_angle < 210) or (right_elbow_angleYZ > 160 and right_elbow_angleYZ < 190)):
            #provera da li su ruke u ravni sa bocnim stranama tela
            if (((landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][2]>landmarks[mp_pose.PoseLandmark.NOSE.value][2] and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][2]<-landmarks[mp_pose.PoseLandmark.NOSE.value][2]) or right_shoulder_angleYZ > 110)
                and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][2]>landmarks[mp_pose.PoseLandmark.NOSE.value][2] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][2]<-landmarks[mp_pose.PoseLandmark.NOSE.value][2])):
            #if right_shoulder_angleYZ >= 120 and right_shoulder_angleYZ < 150:
                #provera da li je leva ruka uz telo
                if left_shoulder_angle > 0 and left_shoulder_angle < 30:
                    #provera da li je desna ruka odrucena ispravno za pokazivanje auta i prekrsaja
                    if right_shoulder_angle > 100 and right_shoulder_angle < 145:
                        label="Signalizirana desna strana."
                    # provera da li je desna ruka odrucena ispravno za pokazivanje udarca iz ugla
                    if right_shoulder_angle > 30 and right_shoulder_angle < 75:
                        label = "Signaliziran udarac iz ugla."
                    # provera da li je desna ruka odrucena ispravno za pokazivanje ofsajda
                    if right_shoulder_angle > 170 and right_shoulder_angle < 190:
                        label = "Signaliziran prekid desnom rukom."
                # provera da li je desna ruka uz telo
                if right_shoulder_angle > 0 and right_shoulder_angle < 30:
                    # provera da li je leva ruka odrucena ispravno za pokazivanje auta i prekrsaja
                    if left_shoulder_angle > 100 and left_shoulder_angle < 145:
                        label = "Signalizirana leva strana."
                    if left_shoulder_angle > 170 and left_shoulder_angle < 185:
                        label = "Signaliziran prekid levom rukom."
                if right_shoulder_angle > 150 and right_shoulder_angle < 185 and left_shoulder_angle > 150 and left_shoulder_angle < 185:
                    label = "Signalizirana zamena."


            #offside
            # provera da li je leva ruka uz telo za offside
            if (left_shoulder_angle > 0 and left_shoulder_angle < 30 and right_shoulder_angleYZ <120
                    and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value][2]<landmarks[mp_pose.PoseLandmark.NOSE.value][2]
                    and (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][2]>landmarks[mp_pose.PoseLandmark.NOSE.value][2] and landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value][2]<-landmarks[mp_pose.PoseLandmark.NOSE.value][2])):
                # provera da li je desna ruka odrucena ispravno za pokazivanje ofsajda u daljini
                if right_shoulder_angleYZ > 92 and right_shoulder_angleYZ < 110:
                    label = "Signaliziran daleki ofsajd."
                # provera da li je desna ruka odrucena ispravno za pokazivanje ofsajda u daljini
                if right_shoulder_angleYZ > 30 and right_shoulder_angleYZ < 85:
                    label = "Signaliziran ofsajd u blizini."
                # provera da li je desna ruka odrucena ispravno za pokazivanje ofsajda u daljini
                if right_shoulder_angleYZ > 85 and right_shoulder_angleYZ < 92:
                    label = "Signaliziran ofsajd na sredini."

    return output_image, label

def isHandStill(list, flag_in_hand):
    if flag_in_hand == "left":
    # racunanje pozicije levog ramena u odnosu na lakat-rame-kuk(ugao leve ruke u odnosu na telo)
        left_shoulder_angle = calculateAngle(list[len(list)-1][mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             list[len(list)-1][mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                             # problem je sto kuk i rame nisu u liniji, a potreban mi je taj ugao da bude 0 kako bih racuna pravi ugao ruke u odnosu na zemlju, ne u odnosu na kuk
                                             (list[len(list)-1][mp_pose.PoseLandmark.LEFT_SHOULDER.value][0],
                                              list[len(list)-1][mp_pose.PoseLandmark.LEFT_HIP.value][1],
                                              list[len(list)-1][mp_pose.PoseLandmark.LEFT_HIP.value][2]))
        if left_shoulder_angle<160:
            return False
        left_shoulder_angleYZ = calculateAngleYZ(list[len(list) - 1][mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                 list[len(list) - 1][mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                 (list[len(list) - 1][mp_pose.PoseLandmark.LEFT_HIP.value][0],
                                                  list[len(list) - 1][mp_pose.PoseLandmark.LEFT_HIP.value][1],
                                                  list[len(list) - 1][mp_pose.PoseLandmark.LEFT_SHOULDER.value][2]))
        if left_shoulder_angleYZ < 120:
            return False


    if flag_in_hand=="right":
        right_shoulder_angleYZ = calculateAngleYZ((list[len(list)-1][mp_pose.PoseLandmark.RIGHT_HIP.value][0],
                                                   list[len(list)-1][mp_pose.PoseLandmark.RIGHT_HIP.value][1],
                                                   list[len(list)-1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value][2]),
                                                  list[len(list)-1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                  list[len(list)-1][mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        if right_shoulder_angleYZ<120:
            return False
            # racunanje pozicije desnog ramena u odnosu na lakat-rame-kuk(ugao desne ruke u odnosu na telo)
        right_shoulder_angle = calculateAngle((list[len(list) - 1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0],
                                               list[len(list) - 1][mp_pose.PoseLandmark.RIGHT_HIP.value][1],
                                               list[len(list) - 1][mp_pose.PoseLandmark.RIGHT_HIP.value][2]),
                                              list[len(list) - 1][mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                              list[len(list) - 1][mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        if right_shoulder_angle < 160:
            return False
    pom=False

    two_step_move=[]
    for i in list:
        angle=0
        if flag_in_hand=="right":
            angle= calculateAngle(i[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       i[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                       i[mp_pose.PoseLandmark.RIGHT_PINKY.value])
        else:
            angle=calculateAngle(i[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                  i[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                  i[mp_pose.PoseLandmark.LEFT_PINKY.value])

        two_step_move.append(angle)

    maxx=max(two_step_move)
    minn=min(two_step_move)
    if maxx-minn>15:
        pom=True
    return pom


# folder path
dir_path = 'media'

# list to store files
res = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        pom=path[(len(path)-4):]
        if pom == '.mp4':
            res.append(path)
#print(res)

# Initialize the VideoCapture object to read from the webcam.
result = ""
def start(files):
    result = ""
    for index, i in enumerate(files):
        camera_video = cv2.VideoCapture(i)
        # Initialize a resizable window.
        cv2.namedWindow('Procenjivanje poze', cv2.WINDOW_NORMAL)
        first_move = False
        first_move_label = ''
        second_move = False
        second_move_label = ''
        move_meaning = ''
        frame_counter = 0
        frame_counter_label = ''
        last_frame_counter = 0
        frame_check_count = 0
        frame_list = []
        two_step_move = False
        two_step_move_attempt = False
        flag_in_hand = "right"
        handNotStill = False
        result+= "\n"+ str(index) +". "+ i.rsplit('/', 1)[-1]+" : "
        # Iterate until the webcam is accessed successfully.
        while camera_video.isOpened():
            # Read a frame.
            ok, frame = camera_video.read()

            # Check if frame is not read properly.
            if not ok:
                # Continue to the next iteration to read the next frame and ignore the empty camera frame.
                continue

            # Flip the frame horizontally for natural (selfie-view) visualization.
            #frame = cv2.flip(frame, 0)
            #frame = cv2.flip(frame, 1)

            # Get the width and height of the frame
            frame_height, frame_width, _ = frame.shape

            # Resize the frame while keeping the aspect ratio.
            frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

            # Detekcija poze
            frame, landmarks = detectPose(frame, pose)

            # Check if the landmarks are detected.
            label=''
            if landmarks:
                # Perform the Pose Classification.
                frame, label = classifyPose(landmarks, frame)

            if not first_move and label=="Signaliziran ofsajd na sredini.":
                label="Signaliziran udarac sa vrata."

            if label == "Signaliziran prekid desnom rukom." or label == "Signaliziran prekid levom rukom.":
                two_step_move = True
                two_step_move_attempt=True
                if label == "Signaliziran prekid levom rukom.":
                    flag_in_hand = "left"
                if label == "Signaliziran prekid desnom rukom.":
                    flag_in_hand = "right"
            if two_step_move:
                frame_check_count += 1
                frame_list.append(landmarks)
            if frame_check_count == 20:
                handNotStill = isHandStill(frame_list, flag_in_hand)
                two_step_move = False
                frame_check_count = 0
                frame_list.clear()

            if handNotStill:
                if flag_in_hand == "right":
                    label = "Signaliziran prekrsaj desnom rukom."
                if flag_in_hand == "left":
                    label = "Signaliziran prekrsaj levom rukom."


            if label != 'Nepoznata poza' and label != '':
                if frame_counter_label != label:
                    frame_counter_label = label
                    frame_counter=1
                if frame_counter_label == label:
                    frame_counter+= 1
                # Opening image
                #--------------------------------------------------------------------------------------------------------------------
                if frame_counter>=20:#ako se frejm ponovi vise od 20 puta znaci da je to konkretan potez, a ne samo pomeraj ruke
                    if not first_move and label!="Signaliziran daleki ofsajd." and label!="Signaliziran ofsajd u blizini." and\
                            label!="Signaliziran ofsajd na sredini.":
                        first_move=True
                        first_move_label = label
                        two_step_move_attempt = False
                    if first_move and not second_move:
                        if label != first_move_label:
                            if (label == "Signaliziran prekid desnom rukom." or label == "Signaliziran prekid levom rukom."
                                                                             or label == "Signaliziran prekrsaj desnom rukom."
                                                                             or label == "Signaliziran prekrsaj levom rukom."
                                                                            ):

                                first_move_label = label
                            elif (label=="Signaliziran daleki ofsajd." or label=="Signaliziran ofsajd u blizini." or label=="Signaliziran ofsajd na sredini.") and first_move_label!="Signaliziran prekid desnom rukom.":
                                if label=="Signaliziran ofsajd na sredini.":
                                    first_move_label="Signaliziran udarac sa vrata."
                                else:
                                    result += "\n" + first_move_label
                                    first_move=False
                                    second_move=False
                            elif label == "Signalizirana zamena.":
                                result += "\n" + first_move_label
                                first_move = False
                                second_move = False
                            else:
                                second_move = True
                                second_move_label = label
                                if (first_move_label == "Signaliziran prekid desnom rukom." or first_move_label == "Signaliziran prekid levom rukom."
                                                                                           or first_move_label == "Signaliziran prekrsaj desnom rukom."
                                                                                           or first_move_label == "Signaliziran prekrsaj levom rukom."):
                                    if first_move_label == "Signaliziran prekid desnom rukom.":#ovo moze i bez ovog or samo donji ifovi da se izbace jedan tab napred tjt, to kad sredjujem kod bla bla gluposti
                                        if second_move_label == "Signalizirana desna strana.":
                                            move_meaning="Signalizirano ubacivanje za napadajucu stranu"
                                        elif second_move_label == "Signaliziran udarac iz ugla.":
                                            move_meaning = "Signaliziran udarac iz ugla za napadajucu stranu"
                                        elif second_move_label == "Signaliziran ofsajd u blizini.":
                                            move_meaning = "Signaliziran ofsajd u blizini."
                                        elif second_move_label == "Signaliziran daleki ofsajd.":
                                            move_meaning = "Signaliziran udaljeni ofsajd."
                                        elif second_move_label == "Signaliziran ofsajd na sredini.":
                                            move_meaning = "Signaliziran ofsajd na sredini."
                                        else:
                                            result += "\n" + first_move_label
                                            first_move = False
                                            second_move = False
                                    elif first_move_label == "Signaliziran prekid levom rukom.":
                                        if second_move_label == "Signalizirana leva strana.":
                                            move_meaning = "Signalizirano ubacivanje za odbrambenu stranu."
                                        else:
                                            result += "\n" + first_move_label
                                            first_move = False
                                            second_move = False
                                    elif first_move_label == "Signaliziran prekrsaj desnom rukom.":
                                        if second_move_label == "Signalizirana desna strana.":
                                            move_meaning = "Signaliziran prekrsaj za napadajucu stranu."
                                        elif second_move_label == "Signalizirana leva strana.":
                                            move_meaning = "Signaliziran prekrsaj za odbrambenu stranu."
                                        else:
                                            result += "\n" + first_move_label
                                            first_move = False
                                            second_move = False
                                    elif first_move_label == "Signaliziran prekrsaj levom rukom.":
                                        if second_move_label == "Signalizirana leva strana.":
                                            move_meaning = "Signaliziran prekrsaj za odbrambenu stranu"
                                        elif second_move_label == "Signalizirana desna strana.":
                                            move_meaning = "Signaliziran prekrsaj za napadajucu stranu"
                                        else:
                                            result += "\n" + first_move_label
                                            first_move = False
                                            second_move = False
                                else:
                                    if 'ofsajd' not in first_move_label:
                                        result += "\n" + first_move_label
                                    first_move = False
                                    second_move = False


                    if second_move:
                        label_holder = label
                        label = move_meaning
                        if len(result) < len(move_meaning) or result[(-len(move_meaning)):] != move_meaning:
                            result += "\n" + move_meaning
                        if second_move_label != label_holder:
                            first_move = False
                            second_move = False
                    color = (0, 255, 0)
        #----------------------------------------------------------------------------------------
            if label == "Nepoznata poza" or label=='' or frame_counter < 20 :
                if handNotStill:
                    color = (0, 255, 0)
                else:
                    label="Nepoznata poza"
                    color=(0, 0, 255)
            #if first_move and not second_move and label=="Signaliziran ofsajd na sredini.":
                #label="Signaliziran udarac sa vrata"
            if first_move and not second_move and (label=="Signaliziran daleki ofsajd."
                                                   or label=="Signaliziran ofsajd u blizini." or label=="Signaliziran ofsajd na sredini."):
                label="Neuspesan pokusaj ofsajda"
                color = (255, 0, 0)
            if two_step_move_attempt and not first_move:
                label="Neuspesan pokusaj prekida"
                color = (255, 0, 0)
            if (label == "Nepoznata poza" or label == '') and first_move:
                if second_move:
                    label = move_meaning
                    color = (0, 255, 0)
                else:
                    label=first_move_label
                    color=(0,255,0)
            if not first_move and not second_move:
                label = "Nepoznata poza"
                color = (0, 0, 255)





            cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
            cv2.imshow('Procenjivanje poze', frame)

            # Wait until a key is pressed.
            # Retreive the ASCII code of the key pressed

            k = cv2.waitKey(1) & 0xFF

            last_frame_counter += 1

            # Check if 'ESC' is pressed.
            if (k == 27 or last_frame_counter == camera_video.get(cv2.CAP_PROP_FRAME_COUNT)):
                # Break the loop.
                break


        # Release the VideoCapture object and close the windows.
        camera_video.release()
        cv2.destroyAllWindows()
        if first_move and not second_move:
            result += "\n"+first_move_label
    f = open("results.txt", "a")
    now = datetime.now()
    current_time = now.strftime("%m/%d/%Y")
    current_time += now.strftime("  %H:%M:%S")
    f.write("\n\n-----I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-I-----"
            "\n\n"+current_time+":\n"+result)
    f.close()
    label_result.configure(text=result)



from tkinter import filedialog

#---------------------------------------------------------------------------------------------------------------
#DESIGN
# Function for opening the
labpom=""

# file explorer window
files = []
def browseFiles():
    global labpom
    filenames = filedialog.askopenfilenames(initialdir = "/", title = "Select a File", filetypes = (("Video zapisi", "*.mp4*"), ("all files", "*.*")))
    # Change label contents

    for i in filenames:
        files.append(i)
        labpom+=i.rsplit('/', 1)[-1]+"\n"
    label_file_explorer.configure(text= labpom)
def deleteFiles():
    global labpom
    files.clear()
    labpom=""
    label_file_explorer.configure(text= labpom)

# Create the root window
window = Tk()

# Set window size
window.geometry("600x500")

window.eval('tk::PlaceWindow . center')
# Set window title
window.title('Prepoznavanje poze fudbalskog sudije pomocnika')

# Set window background color
window.config(background="white")


# Create a File Explorer label
label_file_explorer = Label(window,
                            text="",
                            width=85, height=7,
                            fg="blue")
label_result = Label(window,
                            text="",
                            width=85, height=10,
                            fg="blue")
label_textresult = Label(window,
                            text="Rezultat:",
                            background="white",
                            fg="black")
label_textresult2 = Label(window,
                            text="Odabrani video zapisi:",
                            background="white",
                            fg="black")
button_explore = Button(window,
                        text="Pretrazi",
                        width=15,
                        command=browseFiles)
button_delete = Button(window,
                        text="Obrisi",
                        width=15,
                        command=deleteFiles)

button_start = Button(window,
                        text="Pokreni aplikaciju",
                        width=15,height=2,
                        command= lambda: start(files),
                        background="blue",
                        fg="white")
button_exit = Button(window,
                     text="Izlaz",
                        width=10,
                     command=exit)


# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_textresult2.grid(column=0, row=0, sticky=W)
label_file_explorer.grid(column=0, row=1)

button_explore.grid(column=0, row=2, sticky=E, pady=5)
button_delete.grid(column=0, row=2, sticky=W, pady=5)

button_start.grid(column=0, row=4, pady=20)

label_textresult.grid(column=0, row=5, sticky=W)

button_exit.grid(column=0, row=7, pady=10, sticky=E)

label_result.grid(column=0, row=6)
# Let the window wait for any events
window.mainloop()

