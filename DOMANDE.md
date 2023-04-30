# DOMANDE

- Avendo 2 classi, ha comunque senso adottare un approccio con la class posterio probability, o valutiamo solo il log likelihood ratio ?
- serve fare una validation con un partizionamento fisso (usando la split database del lab5) oppure basta fare la k fold cross validation?
- Dato che le classi sono 2, LDA proietterebbe su m<=1, quindi ha senso non farlo proprio ?
    - LDA approccio 1 si può fare se Sw è definita positiva, lo possiamo assumere o ci poniamo il problema e usiamo l'altro metodo ?

- E' necessario all'inizio plottare i dati per come li carichiamo dal dataset o plottare dopo PCA/LDA ?

- Che valori dare a K nella cross validation? (leave one out immagino sia troppo oneroso come approccio) 


# NOTE / COSE DA FARE
1) Fare 10 istogrammi dei dati caricati feature per feature - - > vedere se si riescono a estrarre informazioni interessanti
2) fare istogrammi su 3 dimensioni plottando a 2 a 2 delle coppie di features per vedere se si possono estrarre info interessanti
3) pca con m=2 per vedere se si riescono a estrarre info interessanti
4) lda per vedere se dall'unica dimensione si riescono a estrarre informazioni interessanti
5) se usiamo LDA:
    - LDA approccio 1 si può fare se Sw è definita positiva, lo possiamo assumere o ci poniamo il problema e usiamo l'altro metodo ?
6) fare pca anche del test set
7) implementare k fold cross validation provando k=5,10 e 2325 (leave one out)
8) FAI PCA ANCHE DEL TEST SET
9) PLOTTA ISTOGRAMMI OF THE DATASET FEATURES (TRAINING SET) DOPO LA LOAD DEI DATI.
