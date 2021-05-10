import os
import numpy as np
import cv2 as cv
import argparse

default_file_suffix='_processed'
org_filename='C:/Users/Dylan/Downloads/video_2.mp4'

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default=os.path.splitext(org_filename)[0]+default_file_suffix, 
    help='set output filename')
args = parser.parse_args()
out_filename=args.output
cap = cv.VideoCapture(org_filename)
fps= int(cap.get(cv.CAP_PROP_FPS))
resol=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH )), 
        int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
if cap.isOpened() is False:
    print("Cannot read video file")
    exit()
myvideo=cv.VideoWriter(out_filename+".mp4", cv.VideoWriter_fourcc('M','P','4','V'), fps, resol)
fgbg = cv.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    if ret==True:
        fgmask = fgbg.apply(frame)
        #cv.imshow('frame',fgmask)
        fg = cv.copyTo(frame,fgmask)
        myvideo.write(fg)
        #cv.imshow('Foreground',fg)
        #cv.imshow('Background',cv.copyTo(frame,cv.bitwise_not(fgmask)))
        k = cv.waitKey(1) & 0xff
        if k & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
myvideo.release()
print('conversion completed.')
cv.destroyAllWindows()