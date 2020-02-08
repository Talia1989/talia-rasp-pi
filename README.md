# talia-rasp-pi

# FRopenCV.py
è il primo programma --> opencv
demo riconoscimento facciale con openCV.
utiliza l'algoritmo "haarcascade_frontalface_default" per riconoscere un volto all'interno dell'immagine.
Elaborazione dell'immagine: l'immagine viene convertita in scala di grigi per far lavoare il classificatore "haarcascade_frontalface_default". Esso è in grado di riconoscere un volto "contando" i pixel contenuti nell'immagine e ragionando sulle differenze di gradazione tra pixel vicini.
Addestramento: l'algoritmo utilizza "face_recognizer" per riconoscere un volto sfruttando il machine learning (creazione cartelle con volti, addestramento modello, ecc..)

# server
tecnologie utilizzate: 
1) libreria Dlib: riconosce i volti tramite "landmarks", cioè con dei punti prefissati che permettono di contornare un viso (inclusi occhi, labbra, naso)
2) openCv e numpy: per elaborazione numerica e d'immagini
3) Flask: per instanziare "server.py"

attraverso openCV si elaborano le immagini catturate dalla "piCamera" e vengono impostati dei valori di soglia con le "trackbar" per agevolare l'algoritmo di riconoscimento (treshold, ecc..).
la combinazione di Dlib, openCV e numpy permette di catturare i frame di un occhio.
attraverso un'opportuna elaborazione d'immagini con openCv si riesce a determinare la direzione dello sguardo.
tutto ciò produce come risultato una stringa nel dominio: {"LEFT", "BLINK","RIGHT", "CENTER"}
attraverso flask la stringa viene esposta all'url "http://..../stream".

# client
tecnologie utilizzate:
1) javascript, html, bootstrap(css): per la presentazione e gestione delle chiamate ai server
2) Spotify Web Playback SDK: per pilotare l'esecuzione delle tracce di spotify (è il passo legato all'automazione)
