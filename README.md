# Face recognition bundle:
Insieme di tre programmi di face recognition in python che sfruttano la libreria opencv per la gestione dell'input video e la libreria face-recognition (basata su dlib).
## buildDatabase.py:
Questo programma crea un file con le codifiche dei volti in un database. Dopo aver aperto il programma fornire un percorso ad un database e aspettare che venga creato un file ".coim" (codified image).
## photoRecognition.py:
Sfrutta il file .coim creato da buildDatabase.py per cercare di riconoscere e dare un nome ad un volto in una foto. Per utilizzarlo è necessario fornire il percorso della foto che si vuole analizzare.
## videoRecognition.py:
Sfrutta il file .coim creato da buildDatabase.py per cercare di riconoscere e dare un nome ad un volto nel live feed di una fotocamera. Per utilizzarlo basta aprirlo.
