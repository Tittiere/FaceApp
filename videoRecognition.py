import face_recognition, pickle, cv2, os

# trovo il path con il file con l'algoritmo per il riconoscimento facciale
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# carico l'algoritmo nel classificatore
faceCascade = cv2.CascadeClassifier(cascPathface)
# do il path al file di encodings
encPath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "daf.coim" #sarebbe encodings.coim
# encPath = os.getcwd() + os.path.sep + "encodings.coim"
# carico i dati dei volti in una variabile
data = pickle.loads(open(encPath, "rb").read())

# apro la telecamera
print("Streaming started")
cv2.useOptimized()
video_capture = cv2.VideoCapture(1)
sicurezza = []
presenze = []
# ciclo che viene eseguito una volta per ogni frame
while True:  
    # considero il frame attuale
    ret, frame = video_capture.read()
    # cambio in scala di grigi e do all'algoritmo
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

    # converto il profilo colore da BGR a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # salvo la codifica del volto
    encodings = face_recognition.face_encodings(rgb)
    names = []
    # cerco di riconoscere la faccia/le facce trovate nel frame
    for encoding in encodings:
        # controllo se ho delle corrispondenze nel database di facce
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        # nome da dare se la persona non è riconosciuta
        name = "Unknown"
        # se trovo un match
        if True in matches:
            # trovo la posizione nell'array di quella corrispondenza
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # ciclo negli id per ogni faccia che corrisponde
            for i in matchedIdxs:
                # dopo che ho salvato gli indici delle facce vado a recuperare il nome
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # prendo il nome di chi ha la maggior corrispondenza
            name = max(counts, key=counts.get) 
        # aggiungo il nome alla lista di nomi
        names.append(name)
        # disegno un quadrato attorno alla faccia riconosciuta
        for ((x, y, w, h), name) in zip(faces, names):
            # colore rosso se faccia non riconosciuta
            col = (0, 255, 0)
            if name == 'Unknown':
                col = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.75, col, 2)
        # se un volto è riconosciuto aggiungo il nome corrispondente alla lista di sicurezza
        if name != 'Unknown':
            sicurezza.append(name)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # quando ho riconosciuto 5 volte un volto controllo che almeno il 60% delle volte sia la stessa persona
    # misura di sicurezza per evitare che in caso di errori il programma segni una presenza non vera
    if len(sicurezza) == 5:
        riscontri = []
        for el in sicurezza:
            riscontri.append(sicurezza.count(el))
        vai = False
        for el in riscontri:
            # se sei stato riconosciuto 3 volte su 5
            if el >= 3:
                vai = True
        if vai == True:
            # estrapolo il nome della persona riconosciuta
            nome = sicurezza[riscontri.index(max(riscontri))]
            try:
                # se è già entrato a scuola lo comunico
                presenze.index(nome)
                print(f'{nome} è già entrato')
            except ValueError:
                # se no lo aggiungo alle presenze se non è già entrato
                print(f'Ho riconosciuto {nome}')
                presenze.append(nome)
        sicurezza = []

video_capture.release()
cv2.destroyAllWindows()
# alla fine del programma stampo le presenze a scuola
print(presenze)