#!/usr/bin/python

import sys
import cv2
import time
import argparse


default_filepath='C:/Users/Dylan/Downloads/video_1.mp4'

def res(self):
    try:
        x, y = map(int, self.split('x'))
        return x, y
    except:
        raise argparse.ArgumentTypeError('Resolution must be widthxheight')

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default=default_filepath, 
    help='set filename')
parser.add_argument('-r', '--resolution', dest='resol', type=res, default=-1, 
    help='set [width,height]')    
parser.add_argument('-f', '--fps', default=-1, 
    help='set fps')
parser.add_argument('-m', '--monochrome', default=0, 
    help='set monochrome 0=normal 1=monochrome')

args = parser.parse_args()

# print command line arguments
#if len(sys.argv)>1:
video_file_path=args.name
print('filename: ', video_file_path)
cap= cv2.VideoCapture(video_file_path)


if args.resol==-1:
    display_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    display_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    display_width=args.resol[0]
    display_height=args.resol[1]
print('resolution', (display_width,display_height))

monochrome=args.monochrome

if args.fps==-1:
    fps= int(cap.get(cv2.CAP_PROP_FPS))
    print("The fps of the video is ", fps)
else:
    fps=int(args.fps)


if cap.isOpened() == False:
    print("Error File Not Found")

while cap.isOpened():
    ret,frame= cap.read()

    if ret == True:
        frame=cv2.resize(frame, (display_width, display_height), interpolation =cv2.INTER_AREA)
        if monochrome:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', frame)
        
        time.sleep(1/fps)
        key=cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('p'):
             while (True):
                key = cv2.waitKey(0)

                # check if 'p' is pressed and resume playing
                if (key & 0xFF == ord('p')):
                    break

                # check if 'b' is pressed and rewind video to the previous frame, but do not play
                if (key & 0xFF == ord('b')):
                    cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    print('* At frame #' + str(cur_frame_number))

                    prev_frame = cur_frame_number
                    if (cur_frame_number > 1):
                        prev_frame -= 1

                    print('* Rewind to frame #' + str(prev_frame))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
    else:
        break


cap.release()
cv2.destroyAllWindows()

