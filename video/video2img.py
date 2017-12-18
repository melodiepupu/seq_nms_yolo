import cv2
import os, sys, getopt
import glob

def usage():
    print 'python video2img.py -i <video file>'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:v", ["help", "input="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    file_name = None
    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            file_name = a
        else:
            assert False, "unhandled option"

    if not os.path.exists('input'):
        os.makedirs('input')

    vidcap = cv2.VideoCapture(file_name)
    success,image = vidcap.read()
    count = 0
    while True:
      success,image = vidcap.read()
      if not success:
          break
      print 'Read a new frame: {}'.format(count)
      cv2.imwrite(os.path.join('input', 'frame{}.jpg'.format(count)), image)     # save frame as JPEG file
      count += 1

if __name__ == "__main__":
   main()
