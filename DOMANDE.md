# DOMANDE

- conviene randomizzare il dataset all'inizio oppure dentro la funzione del k-fold ?

# NOTE / COSE DA FARE
1) fare pca dopo il partizionamento del k-fold (usare la stessa matrice P sia per training set che validation set)
2) Verificare che gli error rate si abbassino, dovrebbero essere più bassi e il modello multivariato più basso del naive bayes
3) modificare e utilizzare il log likelihood ratio soltanto
4) randomizzare il dataset per abbassare errore k-fold con k=5

# RISPOSTE
- Avendo 2 classi, ha comunque senso adottare un approccio con la class posterio probability, o valutiamo solo il log likelihood ratio (log likelihood solo conviene)
- serve fare una validation con un partizionamento fisso (usando la split 2to1 del lab5) oppure basta fare la k fold cross validation (k fold è già più robusto di per sè) (solo k fold è meglio)
- LDA approccio 1 si può fare se Sw è definita positiva, lo possiamo assumere o ci poniamo il problema e usiamo l'altro metodo (possiamo assumere sw positiva)
- E' necessario plottare dopo PCA/LDA, dopo pCA con m=2 quindi parecchio ridotto può aver senso ?
Nuove:
- se facciamo il PCA dentro il kfold, dobbiamo farlo anche coi dati di validazione che usiamo ad ogni iterazione? (anche per validazione, usiamo la stessa matrice P sia per training che validation)

# COSE FATTE
1) Fare 10 istogrammi dei dati caricati feature per feature - - > vedere se si riescono a estrarre informazioni interessanti
2) lda per vedere se dall'unica dimensione si riescono a estrarre informazioni interessanti (ci sono info interessanti)
3) fare istogrammi su 3 dimensioni plottando a 2 a 2 delle coppie di features per vedere se si possono estrarre info interessanti (fatto con scatter plot)
4) pca con m=2 per vedere se si riescono a estrarre info interessanti


# Note 04/05/2023
COSE DETTE DAL PROF A LAURA:
LA LDA RIDUCE A 1 DIMENSIONE  POI IL CLASSIFICATORE NON HA SENSO.
IL TIED CI Dà LO STESSO RISULTATO DELL’LDA.
IL NAIVE NON HA SENSO FARLO CON IL FATTO CHE OTTENIAMO UNA FEATURE.
IL FULL COVARIANCE AVREBBE SENSO SOLO SE I SAMPLE SI DISTRIBUISCONO IN DEI RANGE CHE SONO DIVISIBILI.  QUESTO XK LE DECISION RULE SONO QUADRATICHE  SE ABBIAMO CLASSE 0, POI CLASSE 1 E POI DI NUOVO CLASSE 0, NON è COMODO!

COSE DETTE DAL PROF A NOI:
IL PCA NON VA FATTO PRIMA MA DOPO IL CLASSIFICATORE. ???  PRIMA K FOLD E POI IL PCA!
-NON SERVE FARE LO SPLIT2TO1  BASTA IL K-FOLD  DOBBIAMO VALUTARE NOI IN GENERALE SE ABBIAMO ABBASTANZA DATI DA POTER FARE UN PARTIZIONAMENTO FISSO (IL PROF CI HA DETTO CHE SECONDO LUI NON NE ABBIAMO ABBASTANZA)
-FARE LA VERSIONE CON IL LLR ANZICHé IL CASO GENERALE CON LE CLASS POSTERIOR PROBABILITY  QUESTO XK + AVANTI NEL CORSO VEDREMO DELLE METRICHE CHE SI CALCOLANO DAL LOG LIKELIHOOD RATIO.
-POSSIAMO USARE IL PRIMO APPROCCIO PER LDA  SE SW NON FOSSE SEMIDEFINITA POSITIVA CI DAREBBE Già ERRORE L’IMPLEMENTAZIONE!
-PROIETTARE SULLA DIREZIONE DELL’LDA A CHE SERVE? DOBBIAMO TROVARE UNA SUPERFICIE DI SEPARAZIONE  LO FACCIAMO CON I VARI MODELLI.
CON IL TIED APPLICATO SUI CAMPIONI ORIGINALI OTTENIAMO LO STESSO RISULTATO (LA SEPARAZIONE CHE SI EFFETTUA) CHE OTTENIAMO CON L’LDA.
TIED + LDA  è LA STESSA COSA CHE FA IL TIED (IL PROF DICE CHE BASTA SCRIVERE QUESTO NEL REPORT)  POSSIAMO FARLO PURCHé NEL REPORT DICIAMO CHE STIAMO FACENDO LA STESSA COSA (LO FACCIAMO GIUSTO PER MOSTRARE I PASSAGGI INTERMEDI, COME P.E. L’LDA).
-PLOTTANDO I 10 ISTOGRAMMI PER LE 10 FEATURE CHE ABBIAMO, AVEVAMO NOTATO CHE QUESTI AVEVANO UN ANDAMENTO ABBASTANZA VICINO A QUELLO DI UNA GAUSSIANA.
HA SENSO ALLORA Già TRARRE LE PRIME CONSIDERAZIONI E DIRE CHE TUTTO SOMMATO I MODELLI GAUSSIANI SI PRESTERANNO BENE? IL PROF DICE NO, XK NELLE PROSSIME LEZIONI VEDREMO CHE CAMBIERANNO UN PO’ LE COSE.

DOMANI FACCIAMO VEDERE GLI ISTOGRAMMI CHE ABBIAMO OTTENUTO AL PROF!
