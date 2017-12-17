import cv2
import os, sys, getopt
import glob

def usage():
    print 'python img2video.py -i <image folder>'

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:v", ["help", "input="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    image_folder = None
    verbose = False
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-i", "--input"):
            image_folder = a
        else:
            assert False, "unhandled option"

    if not os.path.exists('output'):
        print 'no output detected'
        sys.exit(2)

    video_name = 'output.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    count = 0
    for element in os.listdir('output'):
        if(str(element).endswith('jpg')):
            count = count + 1
    images = ['frame{}.jpg'.format(i) for i in range(count)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
   main()
