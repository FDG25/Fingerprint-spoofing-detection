# DOMANDE

- Avendo 2 classi, ha comunque senso adottare un approccio con la class posterio probability, o valutiamo solo il log likelihood ratio ?
- serve fare una validation con un partizionamento fisso (usando la split 2to1 del lab5) oppure basta fare la k fold cross validation (k fold è già più robusto di per sè) ?
- LDA approccio 1 si può fare se Sw è definita positiva, lo possiamo assumere o ci poniamo il problema e usiamo l'altro metodo ?
- E' necessario plottare dopo PCA/LDA, dopo pCA con m=2 quindi parecchio ridotto può aver senso ?
Nuove:
- se facciamo il PCA dentro il kfold, dobbiamo farlo anche coi dati di validazione che usiamo ad ogni iterazione? 

# NOTE / COSE DA FARE
1) fare pca anche del test set

# COSE FATTE
1) Fare 10 istogrammi dei dati caricati feature per feature - - > vedere se si riescono a estrarre informazioni interessanti
2) lda per vedere se dall'unica dimensione si riescono a estrarre informazioni interessanti (ci sono info interessanti)
3) fare istogrammi su 3 dimensioni plottando a 2 a 2 delle coppie di features per vedere se si possono estrarre info interessanti (fatto con scatter plot)
4) pca con m=2 per vedere se si riescono a estrarre info interessanti
