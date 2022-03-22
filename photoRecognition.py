import face_recognition, pickle, cv2, os

# trovo il path con il file con l'algoritmo per il riconoscimento facciale
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# carico l'algoritmo nel classificatore
faceCascade = cv2.CascadeClassifier(cascPathface)
# do il path al file di encodings
encPath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "encodings.coim"
# encPath = os.getcwd() + os.path.sep + "encodings.coim"
# carico i dati dei volti in una variabile
data = pickle.loads(open(encPath, "rb").read())


# chiedo all'utente il path della foto da analizzare
# e rimuovo le virgolette in caso ci siano
# ciclo while in caso dia un path non valido
while True:
    path = input('Inserisci il path alla foto da analzizare:\n')
    if path.startswith('"') or path.startswith("'"):
        path = path[1:-1]
    try:
        # col path all'immagine traccio il profilo colori dei pixel
        image = cv2.imread(path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        break
    except:
        pass

# converto l'immagine in scala di grigi per la funzione di riconoscimento
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(60, 60),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
 
# carico la faccia trovata nell'immagine
encodings = face_recognition.face_encodings(rgb)
names = []
# controllo se la faccia riconosciuta corrisponde alle facce salvate nel database
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
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
    # stampo l'immagine
    cv2.imshow("Frame", image)
    cv2.waitKey(0)