# DOMANDE
- CONTROLLARE SE EFFECTIVE PRIOR VA BENE X COME L'ABBIAMO INSERITA NEI GENERATIVE E NEI DISCRIMINATIVE MODELS. --> X COME ABBIAMO FATTO CON LE EFFECTIVE PRIORS OTTENIAMO ERROR RATE TANTO + ALTI --> ARRIVIAMO AD AVERE ERROR RATE FINO AL 33% per il tied naive bayes
- AVEVAMO INIZIALMENTE RANDOMIZZATO IL DATASET PER INTERO MISCHIANDO A CASO TUTTE E 2 LE CLASSI. ADESSO ABBIAMO PROVATO A RANDOMIZZARE SEPARATAMETE I CAMPIONI DELLE 2 CLASSI E POI A STACKARLI IN PROPORZIONE 2:1, VISTO CHE LA CLASSE 0 HA IL DOPPIO DEI CAMPIONI DELLA CLASSE 1 --> COSì ABBIAMO ALCUNI ERROR RATE CHE SONO MIGLIORATI E ALTRI CHE SONO PEGGIORATI, MENTRE L'MVG E IL NAIVE BAYES VENGONO UGUALI. CHE DOBBIAMO FARE?

# NOTE / COSE DA FARE
TESTARE PCA SOTTO AL 7 (A PARTIRE DA 6 CON TUTTI E 7 I VALORI DI LAMBDA CHE ABBIAMO VISTO

# RISPOSTE
- Avendo 2 classi, ha comunque senso adottare un approccio con la class posterio probability, o valutiamo solo il log likelihood ratio (log likelihood solo conviene)
- serve fare una validation con un partizionamento fisso (usando la split 2to1 del lab5) oppure basta fare la k fold cross validation (k fold è già più robusto di per sè) (solo k fold è meglio)
- LDA approccio 1 si può fare se Sw è definita positiva, lo possiamo assumere o ci poniamo il problema e usiamo l'altro metodo (possiamo assumere sw positiva)
- E' necessario plottare dopo PCA/LDA, dopo pCA con m=2 quindi parecchio ridotto può aver senso ?
Nuove:
- se facciamo il PCA dentro il kfold, dobbiamo farlo anche coi dati di validazione che usiamo ad ogni iterazione? (anche per validazione, usiamo la stessa matrice P sia per training che validation)
- Nel progetto anziché fare logistic regression multiclass
- K va bene 5
- Shuffle per come lo abbiamo fatto va bene
- conviene randomizzare il dataset all'inizio oppure dentro la funzione del k-fold (conviene all'inizio)


- con PCA (applicato dentro il K-Fold) notiamo un leggero miglioramento (sia con 5 che con 8), è plausibile ?
- come PCA, anche LDA dev'essere fatto all'interno del k-fold ?
- siccome PCA lo abbiamo fatto nel k-fold, i plot PCA/LDA prima del k-fold applicati sui dati di partenza hanno comunque senso ? 
- RISOLTO DA NOI: z normalization dà errori di 30-40% (provando singolarmente a centrare i dati, standardizzarli e a fare tutte e 2 le cose insieme con la z normalization) --> AVEVAMO SCORDATO DI TRASFORMARE ANCHE IL TEST SET.

# COSE FATTE
1) Fare 10 istogrammi dei dati caricati feature per feature - - > vedere se si riescono a estrarre informazioni interessanti
2) lda per vedere se dall'unica dimensione si riescono a estrarre informazioni interessanti (ci sono info interessanti)
3) fare istogrammi su 3 dimensioni plottando a 2 a 2 delle coppie di features per vedere se si possono estrarre info interessanti (fatto con scatter plot)
4) pca con m=2 per vedere se si riescono a estrarre info interessanti
5) randomizzare il dataset per abbassare errore k-fold con k=5
6) fare pca dopo il partizionamento del k-fold (usare la stessa matrice P sia per training set che validation set)
7) Verificare che gli error rate si abbassino, dovrebbero essere più bassi e il modello multivariato più basso del naive bayes

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

# NOTE 18/05/2023 (RISPOSTE DATE DAL PROF)
1) Nel Logistic regression quadratico, che funzione bisogna passare al minimizzatore? La stessa del caso lineare, solo che passiamo fi(x) anziché x. 
2) Come facciamo a scegliere un valore di lambda opportuno per logistic regression? (Provare logistic regression con valori valori di lambda e vari valori di PCA) 
Proviamo con pca alto vari valori di lambda. Poi proviamo questi valori di lambda. Se vediamo che con pca 8 e pca 7 otteniamo più o meno gli stessi risultati (comunque non migliorano), il pca non vale la pena usarlo 
3) Domanda slide 28-29 BLOCCO 7 - - > visto che nel progetto abbiamo usato una versione regolarizzata del modello (che non è invariante rispetto a trasformazioni lineari dei campioni), non dobbiamo porci il problema di effettuare strategie di preprocessing come centrare i dati, standardizzare le varianze, normalizzare, etc...?
Con mvg non ha senso farlo (NON HA SENSO APPLICARE STRATEGIE DI PREPROCESSING), con logistic regression invece vale la pena provare. 
4) Come tenere in considerazione i costi? Sia per i generative models che per i discriminative models (logistic regression con la formula pesata, in cui compaiono le priors probability), ci basta calcolare pi tilde guardando la formula sulle slide (blocco 8 slide 19)  - - > per problemi binari possiamo sempre far convergere l'applicazione reale a quella con costi unitari (possiamo sempre ragionare come se i costi fossero 1 indipendentemente dal modello)
Sostanzialmente possiamo sempre effettuare questa trasformazione e ricondurci a costi identici (unitari) 

# NOTE 19/05/2023 (RISPOSTE DATE DAL PROF)
- AVENDO FATTO LR WEIGHTED PER CASO LINEARE E QUADRATICO, POSSIAMO FARE A MENO DI USARE LA FORMULA NORMALE CHE AVEVAMO USATO NEL LAB CHE NON TIENE CONTO DELLE PRIOR? BASTA USARE QUELLA PESATA --> QUELLA ORIGINALE (QUELLA CHE AVEVAMO USATO AL LAB7, CHE NON TIENE CONTO DEI PESI) SI PUò OTTENERE SEMPLICEMENTE CONSIDERANDO piT = 0.5 E nt=nf
- SE NOTIAMO CHE PER ALCUNI MODELLI IL PCA CONVIENE MENTRE PER ALTRI NO, POSSIAMO CONFRONTARE TUTTI I VARI MODELLI USANDO PER ALCUNI IL PCA E PER ALTRI NO? MVG E PCA + MVG COSì COME LR E PCA + LR SONO MODELLI SEPARATI --> A NOI INTERESSA QUELLO CHE PERFORMA MEGLIO! 
- Whitening transformation --> come facciamo A*xi --> A è 2x2 mentre xi è (10,1) --> A NON è 2X2 MA è 10X10, DOVE 10 è IL NUMERO DI FEATURES! --> PERTANTO LE DIMENSIONI SONO COMPATIBILI! --> HA DETTO CHE PER CALCOLARE LA MATRICE DI COVARIANZA ^-1/2 DOBBIAMO USARE SVD, MA CI SONO ALTRI MODI CHE PERò NON HA DETTO.

- DETTO A LAURA DAL PROF: NOI AVEVAMO RANDOMIZZATO IL DATASET DI PARTENZA A CASO, MA è MEGLIO RANDOMIZZARE PRIMA TUTTI I CAMPIONI DELLA CLASSE 0, POI I CAMPIONI DELLA CLASSE 1, DOPODICHé ANDARE A COSTRUIRE A POCO A POCO IL NUOVO DATASET RANDOMIZZANDO PRENDENDO 2 CAMPIONI DAL SET RANDOMIZZATO DELLA CLASSE 0 E 1 CAMPIONE DAL SET RANDOMIZZATO DELLA CLASSE 1 (DOVE PER CLASSE 0 INTENDIAMO QUELLA PER CUI ABBIAMO + CAMPIONI IN PROPORZIONE 2/3)


# ROBE LATEX

\begin{table}
\centering
\begin{tabular}{ccc}
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature2.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature3.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature4.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature5.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature6.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature7.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature8.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature1 Feature10.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature3.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature4.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature5.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature6.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature7.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature8.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature2 Feature10.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature4.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature5.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature6.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature7.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature8.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature3 Feature10.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature4 Feature5.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature4 Feature6.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature4 Feature7.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature4 Feature8.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature4 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature4 Feature10.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature5 Feature6.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature5 Feature7.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature5 Feature8.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature5 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature5 Feature10.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature6 Feature7.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature6 Feature8.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature6 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature6 Feature10.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature7 Feature8.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature7 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature7 Feature10.png}} \\
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature8 Feature9.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature8 Feature10.png}} &
\subfloat{\includegraphics[width = 1.3in]{ScatterPlots/Feature9 Feature10.png}} \\

\end{tabular}
\caption{Histogram of the given dataset features (training set). 
Red histograms refer to authentic fingerprints, blue histograms to spoofed fingerprints.}
\end{table}
