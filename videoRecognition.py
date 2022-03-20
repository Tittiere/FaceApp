import face_recognition
import pickle
import cv2
import os

#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

# cascPath = 'G:\\Il mio Drive\\ICT\\python\\robo vision\\face recon\\Il tipo ez\\haarcascade_frontalface_default.xml'
# faceCascade = cv2.CascadeClassifier(cascPath)

encPath = os.getcwd() + os.path.sep + "encodings.coim"
# encPath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "encodings.coim"
# load the known faces and embeddings saved in last file
data = pickle.loads(open(encPath, "rb").read())

print("Streaming started")
cv2.useOptimized()
video_capture = cv2.VideoCapture(0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        #set name =Unknown if no encoding matches
        name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            col = (0, 255, 0)
            if name == 'Unknown':
                col = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, col, 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()