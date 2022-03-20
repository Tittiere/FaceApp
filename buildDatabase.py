import face_recognition, pickle, cv2, os

path = r"C:\Users\matti\Desktop\FOTO SCIENZE\\"
filesInData = os.listdir(path)

knownEncodings = []
knownNames = ['.']

for k in filesInData:
    if k.find('.') == -1:
        dirPath = path + k + '\\'
        filesInDir = os.listdir(dirPath)
        imgPaths = [dirPath + e for e in filesInDir if not (e.endswith('.ini')) and not (e.endswith('.py')) and not (e.endswith('.xml'))]
        for (i, el)  in enumerate(imgPaths):
            name = el.split(os.path.sep)[-2]
            if len(name.split('_')) == 3:
                name = name.split('_')
                name = name[0] + ' ' + name[2]
            print(name)
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(el)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(rgb,model='hog')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
knownNames.pop(0)
conto = 0
for i in range(len(knownNames)):
    if i != 0:
        if knownNames[i] != knownNames[i-1]:
            conto += 1
    else:
        conto += 1

print(f'conto = {conto}')
#save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
#use pickle to save data into a file for later use
facce = os.path.dirname(os.path.realpath(__file__)) + '\\' + "final_enc"
f = open(facce, "wb")
f.write(pickle.dumps(data))
f.close()