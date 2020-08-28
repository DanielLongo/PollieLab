import cv2
import imutils
import numpy as np


def main():
    red = [0, 0, 255]
    cap = cv2.VideoCapture(0)
    greenLower = (0, 0, 0)
    greenUpper = (20, 20, 20)
    min_radius = 5
    pollies = []
    num_pollies = 5
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = frame

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)


        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for cnt in cnts:
            c = cnt
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > min_radius:
                pollies.append((radius, center))

        if len(pollies) != num_pollies:
            print(len(pollies), "found but there are only supposed to be", num_pollies)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        for pollie in pollies:
            center = pollie[1]
            frame[center[0] - 5 : center[0] + 5, center[1] - 5: center[1] + 5] = red
        # Display the resulting frame
        cv2.imshow('frame', frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
