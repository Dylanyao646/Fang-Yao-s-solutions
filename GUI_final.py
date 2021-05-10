import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import sys
import argparse

default_filepath='C:/Users/Dylan/Downloads/video_2.mp4'

def res(self):
    try:
        x, y = map(int, self.split('x'))
        return x, y
    except:
        raise argparse.ArgumentTypeError('Resolution must be widthxheight')

class App:
    def __init__(self, window, window_title, video_file_path, param):
        self.window = window
        self.window.title(window_title)
        self.video_file_path = video_file_path
        self.paused=0

        self.window.bind('q', lambda event: self.quit())
        self.window.bind('p', lambda event: self.pause())
        self.window.bind('b', lambda event: self.stepback())
        # open video
        self.vid = MyVideoCapture(self.video_file_path)
        
        if args.resol==-1:
            canvas_width=self.vid.width
            canvas_height=self.vid.height
        else:
            canvas_width=args.resol[0]
            canvas_height=args.resol[1]
        self.resol=(canvas_width,canvas_height)
        print('resolution:', (canvas_width,canvas_height))
        if args.fps==-1:
            self.fps=self.vid.fps
        else:
            self.fps=int(args.fps)
        print('fps:', self.fps)
        self.monochrome=args.monochrome
        print('monochrome:', self.monochrome)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = canvas_width, height = canvas_height)
        self.canvas.pack()

        # Button that lets the user to pause the video
        self.btn_pause=tkinter.Button(window, text="Pause", width=50, command=self.pause)
        self.btn_pause.pack(anchor=tkinter.CENTER, expand=True)
        
        # Button that lets the user to reverse frame
        self.btn_stepback=tkinter.Button(window, text="Reverse Frame", width=50, command=self.stepback)
        self.btn_stepback.pack(anchor=tkinter.CENTER, expand=True)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay=int(1/self.fps*1000)
        self.update()
        self.window.mainloop()

    def pause(self):
        if not(self.paused):
            self.paused=1
            print('paused')
        else:
            self.paused=0
            print('resume')
        self.window.after(self.delay, self.update)
        

            # check if 'r' is pressed and rewind video to frame 0, then resume playing
            #if (key & 0xFF == ord('r')):
            #    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            #    break
    def stepback(self):
        cur_frame_number = self.vid.get_cur_frame_num()
        print('* At frame #' + str(cur_frame_number))
        prev_frame = cur_frame_number
        if (cur_frame_number > 1):
            prev_frame -= 1
        print('* Rewind to frame #' + str(prev_frame))
        self.vid.set_frame_num(prev_frame)

    def update(self):
        # Get a frame from the video source
        if not(self.paused):
            ret, frame = self.vid.get_frame(self.monochrome, self.resol)
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.window.after(self.delay, self.update)
        else:
            pass

    def quit(self):
        print('quiting')
        del self.vid
        self.window.destroy()


class MyVideoCapture:
    def __init__(self, video_file_path):
        # Open the video source
        self.vid = cv2.VideoCapture(video_file_path)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_file_path)

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps= int(self.vid.get(cv2.CAP_PROP_FPS))

    def get_frame(self,monochrome,resol):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame=cv2.resize(frame, resol, interpolation =cv2.INTER_AREA)
                # Return a boolean success flag and the current frame converted to BGR
                if monochrome:
                    return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                else:
                    return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    def get_cur_frame_num(self):
        if self.vid.isOpened():
            cur_frame_number=self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            return cur_frame_number
        else:
            return None
    def set_frame_num(self,frame_num):
        if self.vid.isOpened():
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


# parse command line arguments
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

# print filename
video_file_path=args.name
print('filename: ', video_file_path)

    
# Create a window and pass it to the Application object

App(tkinter.Tk(), "Tkinter and OpenCV", video_file_path, args)
