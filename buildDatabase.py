import face_recognition, pickle, time, cv2, os

# funzione per pulire la shell
def clear():
    # scelgo il comando in base al sistema operativo
    if str(os.path.sep) == '/':
        os.system('clear')
    else:
        os.system('cls')

# chiedo all'utente il path della cartella che contiene il
# database e rimuovo le virgolette in caso ci siano
path = input('Inserisci il path alla cartella contenente il database:\n')
time.sleep(1.5)
if path.startswith('"') or path.startswith("'"):
    path = path[1:-1]
# aggiungo il separatore path di sistema
path += os.path.sep
# trovo, in ordine alfabetico, i nomi di tutto quello
# che è contenutonella cartella del database
filesInData = os.listdir(path)

# inizializzo gli array che conterranno i valori dei volti
# codificati ed i nomi corrispettivi
knownEncodings = []
knownNames = []

# algoritmo inutilmente complicato per visualizzare il progresso
# è esattamente lo stesso procedimento spiegato dopo, serve a contare
# quante immagini ci sono per stampare un progresso nel ciclo dopo
totEls = 0
for a in filesInData:
    if a.find('.') == -1:
        dirPath = path + a + os.path.sep
        filesInDir = os.listdir(dirPath)
        elements = [dirPath + e for e in filesInDir if not (e.endswith('.ini')) and not (e.endswith('.py')) and not (e.endswith('.xml'))]
        for b in elements:
            totEls += 1

# inizializzo una variabile per contare le immagini
count = 0
# per ogni elemento k nei files nel database
for k in filesInData:
    # se l'elemento k ha un punto nel nome significa che non è
    # una cartella. Se è una cartella allora:
    if k.find('.') == -1:
        # creo il path della cartella aggiungendo il nome della cartella al path
        dirPath = path + k + os.path.sep
        # ottengo le immagini nella cartella corrispondente al nome
        filesInDir = os.listdir(dirPath)
        # creo un array con i precorsi integrali di tutte le immagini nella cartella
        # escludendo eventuali file di config della cartella o file python
        imgPaths = [dirPath + e for e in filesInDir if not (e.endswith('.ini')) and not (e.endswith('.py')) and not (e.endswith('.xml'))]
        # ciclo for che scorre le immagini percorso per percorso numerandole
        for (i, el)  in enumerate(imgPaths):
            # pulisco la shell
            clear()
            # stampo il progresso
            count += 1
            print(f'{count}/{totEls}')
            # carico l'immagine e la converto da BGR a un formato
            # che la libreria "dlib" possa usare (RGB)
            image = cv2.imread(el)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # utilizzo l'algoritmo di face recognition per trovare le facce
            boxes = face_recognition.face_locations(rgb,model='hog')
            # codifico il volto e lo salvo in una variabile
            encodings = face_recognition.face_encodings(rgb, boxes)
            # scorro l'array con le codifiche in caso più volti vengano trovati
            for encoding in encodings:
                # aggiungo la codifica del volto all'array dove le ho tutte
                knownEncodings.append(encoding)
                # aggiungo il nome corrispondente per riconoscerlo
                knownNames.append(k)
            

# salvo gli encodings in un dizionario insieme ai nomi corrispondenti
data = {"encodings": knownEncodings, "names": knownNames}
# codifico con pickle i dati e li scrivo in un file
facce = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "encodings.coim"
f = open(facce, "wb")
f.write(pickle.dumps(data))
f.close()