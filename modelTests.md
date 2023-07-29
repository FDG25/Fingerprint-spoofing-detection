# Models

- [Gaussians](#gaussians)
- [Logistic Regression](#lr-prior--05)
    - [Best Values Linear](#linear-lr-best-values)
    - [Best Values Quadratic](#quadratic-lr-best-values)
- [Linear SVM](#svm-linear-hyperparameters-k-and-c-training-prior--05)
# Gaussians

## RAW (No PCA No Z_Norm)
### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Multivariate Gaussian Classifier: 0.3314549180327869

Naive Bayes results:
Accuracy: 92.47%
Error rate: 7.53%
Min DCF for Naive Bayes: 0.4716803278688525

Tied Covariance results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Tied Covariance: 0.4861475409836066

Tied Naive Bayes results:
Accuracy: 88.34%
Error rate: 11.66%
Min DCF for Tied Naive Bayes: 0.5507991803278689

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Multivariate Gaussian Classifier: 0.6155327868852459

Naive Bayes results:
Accuracy: 92.47%
Error rate: 7.53%
Min DCF for Naive Bayes: 0.8057991803278688

Tied Covariance results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Tied Covariance: 0.70625

Tied Naive Bayes results:
Accuracy: 88.34%
Error rate: 11.66%
Min DCF for Tied Naive Bayes: 0.7900000000000001

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Multivariate Gaussian Classifier: 0.11057377049180327

Naive Bayes results:
Accuracy: 92.47%
Error rate: 7.53%
Min DCF for Naive Bayes: 0.14440346083788705

Tied Covariance results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Tied Covariance: 0.18448998178506373

Tied Naive Bayes results:
Accuracy: 88.34%
Error rate: 11.66%
Min DCF for Tied Naive Bayes: 0.197825591985428


## Z_Norm

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Multivariate Gaussian Classifier: 0.3577049180327869

Naive Bayes results:
Accuracy: 92.0%
Error rate: 8.0%
Min DCF for Naive Bayes: 0.47862704918032783

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.4843237704918033

Tied Naive Bayes results:
Accuracy: 88.52%
Error rate: 11.48%
Min DCF for Tied Naive Bayes: 0.5511270491803278

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Multivariate Gaussian Classifier: 0.6117827868852459

Naive Bayes results:
Accuracy: 92.0%
Error rate: 8.0%
Min DCF for Naive Bayes: 0.7895491803278689

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.7249999999999999

Tied Naive Bayes results:
Accuracy: 88.52%
Error rate: 11.48%
Min DCF for Tied Naive Bayes: 0.78125

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Multivariate Gaussian Classifier: 0.11150956284153003

Naive Bayes results:
Accuracy: 92.0%
Error rate: 8.0%
Min DCF for Naive Bayes: 0.15328096539162112

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.18313979963570123

Tied Naive Bayes results:
Accuracy: 88.52%
Error rate: 11.48%
Min DCF for Tied Naive Bayes: 0.20303961748633875

## RAW + PCA with M = 10

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Multivariate Gaussian Classifier: 0.3314549180327869

Naive Bayes results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Naive Bayes: 0.36985655737704914

Tied Covariance results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Tied Covariance: 0.4861475409836066

Tied Naive Bayes results:
Accuracy: 89.03%
Error rate: 10.97%
Min DCF for Tied Naive Bayes: 0.5329508196721311

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Multivariate Gaussian Classifier: 0.6155327868852459

Naive Bayes results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Naive Bayes: 0.7223155737704918

Tied Covariance results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Tied Covariance: 0.70625

Tied Naive Bayes results:
Accuracy: 89.03%
Error rate: 10.97%
Min DCF for Tied Naive Bayes: 0.7540163934426228

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Multivariate Gaussian Classifier: 0.11057377049180327

Naive Bayes results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Naive Bayes: 0.11294626593806921

Tied Covariance results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Tied Covariance: 0.18448998178506373

Tied Naive Bayes results:
Accuracy: 89.03%
Error rate: 10.97%
Min DCF for Tied Naive Bayes: 0.1982536429872495


## Z_Norm + PCA with M = 10

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Multivariate Gaussian Classifier: 0.3577049180327869

Naive Bayes results:
Accuracy: 93.89%
Error rate: 6.11%
Min DCF for Naive Bayes: 0.3855122950819672

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.4843237704918033

Tied Naive Bayes results:
Accuracy: 88.69%
Error rate: 11.31%
Min DCF for Tied Naive Bayes: 0.5801844262295082

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Multivariate Gaussian Classifier: 0.6117827868852459

Naive Bayes results:
Accuracy: 93.89%
Error rate: 6.11%
Min DCF for Naive Bayes: 0.6967827868852459

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.7249999999999999

Tied Naive Bayes results:
Accuracy: 88.69%
Error rate: 11.31%
Min DCF for Tied Naive Bayes: 0.77625

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Multivariate Gaussian Classifier: 0.11150956284153003

Naive Bayes results:
Accuracy: 93.89%
Error rate: 6.11%
Min DCF for Naive Bayes: 0.12742486338797812

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.18313979963570123

Tied Naive Bayes results:
Accuracy: 88.69%
Error rate: 11.31%
Min DCF for Tied Naive Bayes: 0.21168943533697632

## RAW + PCA with M = 9

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.37%
Error rate: 5.63%
Min DCF for Multivariate Gaussian Classifier: 0.3295696721311475

Naive Bayes results:
Accuracy: 94.28%
Error rate: 5.72%
Min DCF for Naive Bayes: 0.36862704918032785

Tied Covariance results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Tied Covariance: 0.4917827868852459

Tied Naive Bayes results:
Accuracy: 88.86%
Error rate: 11.14%
Min DCF for Tied Naive Bayes: 0.5429713114754099

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.37%
Error rate: 5.63%
Min DCF for Multivariate Gaussian Classifier: 0.6292827868852459

Naive Bayes results:
Accuracy: 94.28%
Error rate: 5.72%
Min DCF for Naive Bayes: 0.7398155737704918

Tied Covariance results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Tied Covariance: 0.69625

Tied Naive Bayes results:
Accuracy: 88.86%
Error rate: 11.14%
Min DCF for Tied Naive Bayes: 0.771516393442623

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.37%
Error rate: 5.63%
Min DCF for Multivariate Gaussian Classifier: 0.10910974499089252

Naive Bayes results:
Accuracy: 94.28%
Error rate: 5.72%
Min DCF for Naive Bayes: 0.1125318761384335

Tied Covariance results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Tied Covariance: 0.17980418943533694

Tied Naive Bayes results:
Accuracy: 88.86%
Error rate: 11.14%
Min DCF for Tied Naive Bayes: 0.20191029143897993


## Z_Norm + PCA with M = 9

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.15%
Error rate: 5.85%
Min DCF for Multivariate Gaussian Classifier: 0.3764139344262295

Naive Bayes results:
Accuracy: 93.85%
Error rate: 6.15%
Min DCF for Naive Bayes: 0.38895491803278687

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.48870901639344266

Tied Naive Bayes results:
Accuracy: 88.82%
Error rate: 11.18%
Min DCF for Tied Naive Bayes: 0.5867418032786885

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.15%
Error rate: 5.85%
Min DCF for Multivariate Gaussian Classifier: 0.6082991803278689

Naive Bayes results:
Accuracy: 93.85%
Error rate: 6.15%
Min DCF for Naive Bayes: 0.7042827868852459

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.7180327868852457

Tied Naive Bayes results:
Accuracy: 88.82%
Error rate: 11.18%
Min DCF for Tied Naive Bayes: 0.7800000000000001

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.15%
Error rate: 5.85%
Min DCF for Multivariate Gaussian Classifier: 0.11890255009107466

Naive Bayes results:
Accuracy: 93.85%
Error rate: 6.15%
Min DCF for Naive Bayes: 0.12661657559198541

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.18532559198542803

Tied Naive Bayes results:
Accuracy: 88.82%
Error rate: 11.18%
Min DCF for Tied Naive Bayes: 0.21408925318761385

## RAW + PCA with M = 8

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.32%
Error rate: 5.68%
Min DCF for Multivariate Gaussian Classifier: 0.33331967213114755

Naive Bayes results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Naive Bayes: 0.3598975409836066

Tied Covariance results:
Accuracy: 90.8%
Error rate: 9.2%
Min DCF for Tied Covariance: 0.4852459016393443

Tied Naive Bayes results:
Accuracy: 89.25%
Error rate: 10.75%
Min DCF for Tied Naive Bayes: 0.5439139344262296

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.32%
Error rate: 5.68%
Min DCF for Multivariate Gaussian Classifier: 0.6125

Naive Bayes results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Naive Bayes: 0.7123155737704918

Tied Covariance results:
Accuracy: 90.8%
Error rate: 9.2%
Min DCF for Tied Covariance: 0.68625

Tied Naive Bayes results:
Accuracy: 89.25%
Error rate: 10.75%
Min DCF for Tied Naive Bayes: 0.77375

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.32%
Error rate: 5.68%
Min DCF for Multivariate Gaussian Classifier: 0.10879553734061928

Naive Bayes results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Naive Bayes: 0.11201047358834243

Tied Covariance results:
Accuracy: 90.8%
Error rate: 9.2%
Min DCF for Tied Covariance: 0.18126138433515482

Tied Naive Bayes results:
Accuracy: 89.25%
Error rate: 10.75%
Min DCF for Tied Naive Bayes: 0.20148907103825134


## Z_Norm + PCA with M = 8

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.3589344262295082

Naive Bayes results:
Accuracy: 94.02%
Error rate: 5.98%
Min DCF for Naive Bayes: 0.3852049180327869

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.5014959016393443

Tied Naive Bayes results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Tied Naive Bayes: 0.5864344262295083

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.6007991803278688

Naive Bayes results:
Accuracy: 94.02%
Error rate: 5.98%
Min DCF for Naive Bayes: 0.7017827868852459

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.7305327868852459

Tied Naive Bayes results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Tied Naive Bayes: 0.78625

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.12723132969034606

Naive Bayes results:
Accuracy: 94.02%
Error rate: 5.98%
Min DCF for Naive Bayes: 0.12066029143897995

Tied Covariance results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Tied Covariance: 0.18626821493624768

Tied Naive Bayes results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Tied Naive Bayes: 0.21127504553734058

## RAW + PCA with M = 7

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.06%
Error rate: 5.94%
Min DCF for Multivariate Gaussian Classifier: 0.3411270491803279

Naive Bayes results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Naive Bayes: 0.3614549180327869

Tied Covariance results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Tied Covariance: 0.4839754098360656

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.5414139344262295

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.06%
Error rate: 5.94%
Min DCF for Multivariate Gaussian Classifier: 0.6180327868852459

Naive Bayes results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Naive Bayes: 0.7173155737704918

Tied Covariance results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Tied Covariance: 0.6999999999999998

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.7690163934426228

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.06%
Error rate: 5.94%
Min DCF for Multivariate Gaussian Classifier: 0.11452413479052823

Naive Bayes results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Naive Bayes: 0.11471766848816027

Tied Covariance results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Tied Covariance: 0.1833469945355191

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.19887522768670307


## Z_Norm + PCA with M = 7

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.3402254098360656

Naive Bayes results:
Accuracy: 93.85%
Error rate: 6.15%
Min DCF for Naive Bayes: 0.3852049180327869

Tied Covariance results:
Accuracy: 90.49%
Error rate: 9.51%
Min DCF for Tied Covariance: 0.4824385245901639

Tied Naive Bayes results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Tied Naive Bayes: 0.5861270491803279

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.6267827868852458

Naive Bayes results:
Accuracy: 93.85%
Error rate: 6.15%
Min DCF for Naive Bayes: 0.7167827868852459

Tied Covariance results:
Accuracy: 90.49%
Error rate: 9.51%
Min DCF for Tied Covariance: 0.730266393442623

Tied Naive Bayes results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Tied Naive Bayes: 0.7900000000000001

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.12370218579234972

Naive Bayes results:
Accuracy: 93.85%
Error rate: 6.15%
Min DCF for Naive Bayes: 0.12013888888888886

Tied Covariance results:
Accuracy: 90.49%
Error rate: 9.51%
Min DCF for Tied Covariance: 0.18532559198542803

Tied Naive Bayes results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Tied Naive Bayes: 0.21263205828779597

## RAW + PCA with M = 6

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.28%
Error rate: 5.72%
Min DCF for Multivariate Gaussian Classifier: 0.3355122950819672

Naive Bayes results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Naive Bayes: 0.3598975409836066

Tied Covariance results:
Accuracy: 90.84%
Error rate: 9.16%
Min DCF for Tied Covariance: 0.4833606557377049

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.5492213114754099

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.28%
Error rate: 5.72%
Min DCF for Multivariate Gaussian Classifier: 0.6117827868852459

Naive Bayes results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Naive Bayes: 0.7210655737704919

Tied Covariance results:
Accuracy: 90.84%
Error rate: 9.16%
Min DCF for Tied Covariance: 0.7299999999999999

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.7802663934426229

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.28%
Error rate: 5.72%
Min DCF for Multivariate Gaussian Classifier: 0.10910974499089252

Naive Bayes results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Naive Bayes: 0.11596766848816027

Tied Covariance results:
Accuracy: 90.84%
Error rate: 9.16%
Min DCF for Tied Covariance: 0.18063979963570123

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.20065346083788707


## Z_Norm + PCA with M = 6

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 93.59%
Error rate: 6.41%
Min DCF for Multivariate Gaussian Classifier: 0.3645901639344262

Naive Bayes results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Naive Bayes: 0.4023565573770492

Tied Covariance results:
Accuracy: 90.24%
Error rate: 9.76%
Min DCF for Tied Covariance: 0.5076844262295082

Tied Naive Bayes results:
Accuracy: 88.39%
Error rate: 11.61%
Min DCF for Tied Naive Bayes: 0.6107991803278688

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 93.59%
Error rate: 6.41%
Min DCF for Multivariate Gaussian Classifier: 0.7017827868852459

Naive Bayes results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Naive Bayes: 0.7682991803278689

Tied Covariance results:
Accuracy: 90.24%
Error rate: 9.76%
Min DCF for Tied Covariance: 0.7192827868852458

Tied Naive Bayes results:
Accuracy: 88.39%
Error rate: 11.61%
Min DCF for Tied Naive Bayes: 0.7550000000000001

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 93.59%
Error rate: 6.41%
Min DCF for Multivariate Gaussian Classifier: 0.12648907103825136

Naive Bayes results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Naive Bayes: 0.12566029143897994

Tied Covariance results:
Accuracy: 90.24%
Error rate: 9.76%
Min DCF for Tied Covariance: 0.1958401639344262

Tied Naive Bayes results:
Accuracy: 88.39%
Error rate: 11.61%
Min DCF for Tied Naive Bayes: 0.22239754098360653

## RAW + PCA with M = 5

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 94.15%
Error rate: 5.85%
Min DCF for Multivariate Gaussian Classifier: 0.35930327868852463

Naive Bayes results:
Accuracy: 94.41%
Error rate: 5.59%
Min DCF for Naive Bayes: 0.3714549180327869

Tied Covariance results:
Accuracy: 90.41%
Error rate: 9.59%
Min DCF for Tied Covariance: 0.4896106557377049

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.5304713114754098

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 94.15%
Error rate: 5.85%
Min DCF for Multivariate Gaussian Classifier: 0.6515163934426229

Naive Bayes results:
Accuracy: 94.41%
Error rate: 5.59%
Min DCF for Naive Bayes: 0.7742827868852459

Tied Covariance results:
Accuracy: 90.41%
Error rate: 9.59%
Min DCF for Tied Covariance: 0.70625

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.786516393442623

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 94.15%
Error rate: 5.85%
Min DCF for Multivariate Gaussian Classifier: 0.11474499089253187

Naive Bayes results:
Accuracy: 94.41%
Error rate: 5.59%
Min DCF for Naive Bayes: 0.11453096539162112

Tied Covariance results:
Accuracy: 90.41%
Error rate: 9.59%
Min DCF for Tied Covariance: 0.191575591985428

Tied Naive Bayes results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Tied Naive Bayes: 0.2067622950819672


## Z_Norm + PCA with M = 5

### Prior = 0.5
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.3888934426229508

Naive Bayes results:
Accuracy: 93.94%
Error rate: 6.06%
Min DCF for Naive Bayes: 0.39297131147540987

Tied Covariance results:
Accuracy: 89.81%
Error rate: 10.19%
Min DCF for Tied Covariance: 0.5424180327868853

Tied Naive Bayes results:
Accuracy: 88.6%
Error rate: 11.4%
Min DCF for Tied Naive Bayes: 0.6247950819672131

### Prior = 0.1
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.6642827868852459

Naive Bayes results:
Accuracy: 93.94%
Error rate: 6.06%
Min DCF for Naive Bayes: 0.711516393442623

Tied Covariance results:
Accuracy: 89.81%
Error rate: 10.19%
Min DCF for Tied Covariance: 0.742766393442623

Tied Naive Bayes results:
Accuracy: 88.6%
Error rate: 11.4%
Min DCF for Tied Naive Bayes: 0.7975

### Prior = 0.9
Multivariate Gaussian Classifier results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Multivariate Gaussian Classifier: 0.12483834244080144

Naive Bayes results:
Accuracy: 93.94%
Error rate: 6.06%
Min DCF for Naive Bayes: 0.12191712204007284

Tied Covariance results:
Accuracy: 89.81%
Error rate: 10.19%
Min DCF for Tied Covariance: 0.20031876138433513

Tied Naive Bayes results:
Accuracy: 88.6%
Error rate: 11.4%
Min DCF for Tied Naive Bayes: 0.22071265938069215

# LR Prior = 0.5

## RAW (No PCA No Z_Norm)

### lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.47116803278688524

Logistic Regression Weighted Quadratic results:
Accuracy: 93.76%
Error rate: 6.24%
Min DCF for Logistic Regression Weighted Quadratic: 0.3308401639344262

### lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.47116803278688524

Logistic Regression Weighted Quadratic results:
Accuracy: 93.68%
Error rate: 6.32%
Min DCF for Logistic Regression Weighted Quadratic: 0.33209016393442625

### lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.47116803278688524

Logistic Regression Weighted Quadratic results:
Accuracy: 93.51%
Error rate: 6.49%
Min DCF for Logistic Regression Weighted Quadratic: 0.31739754098360656

### lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.23%
Error rate: 12.77%
Min DCF for Logistic Regression Weighted: 0.4736680327868853

Logistic Regression Weighted Quadratic results:
Accuracy: 93.63%
Error rate: 6.37%
Min DCF for Logistic Regression Weighted Quadratic: 0.2923975409836066

### lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.41%
Error rate: 13.59%
Min DCF for Logistic Regression Weighted: 0.4771311475409836

Logistic Regression Weighted Quadratic results:
Accuracy: 93.16%
Error rate: 6.84%
Min DCF for Logistic Regression Weighted Quadratic: 0.30336065573770493

### lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 78.19%
Error rate: 21.81%
Min DCF for Logistic Regression Weighted: 0.4889549180327869

Logistic Regression Weighted Quadratic results:
Accuracy: 92.95%
Error rate: 7.05%
Min DCF for Logistic Regression Weighted Quadratic: 0.31206967213114756

### lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.553360655737705

Logistic Regression Weighted Quadratic results:
Accuracy: 91.87%
Error rate: 8.13%
Min DCF for Logistic Regression Weighted Quadratic: 0.31643442622950824

### lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5823770491803278

Logistic Regression Weighted Quadratic results:
Accuracy: 88.39%
Error rate: 11.61%
Min DCF for Logistic Regression Weighted Quadratic: 0.34737704918032786

## Z_Norm

### lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.05%
Error rate: 12.95%
Min DCF for Logistic Regression Weighted: 0.4852459016393443

Logistic Regression Weighted Quadratic results:
Accuracy: 93.38%
Error rate: 6.62%
Min DCF for Logistic Regression Weighted Quadratic: 0.32709016393442625

### lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 86.97%
Error rate: 13.03%
Min DCF for Logistic Regression Weighted: 0.4852459016393443

Logistic Regression Weighted Quadratic results:
Accuracy: 93.38%
Error rate: 6.62%
Min DCF for Logistic Regression Weighted Quadratic: 0.31772540983606556

### lambda value : 0.001 BEST QUADRATIC
Logistic Regression Weighted results:
Accuracy: 86.49%
Error rate: 13.51%
Min DCF for Logistic Regression Weighted: 0.4868032786885246

Logistic Regression Weighted Quadratic results:
Accuracy: 93.42%
Error rate: 6.58%
Min DCF for Logistic Regression Weighted Quadratic: 0.2852254098360656

### lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 84.09%
Error rate: 15.91%
Min DCF for Logistic Regression Weighted: 0.4974385245901639

Logistic Regression Weighted Quadratic results:
Accuracy: 91.66%
Error rate: 8.34%
Min DCF for Logistic Regression Weighted Quadratic: 0.2976844262295082

### lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.83%
Error rate: 32.17%
Min DCF for Logistic Regression Weighted: 0.5177049180327868

Logistic Regression Weighted Quadratic results:
Accuracy: 81.08%
Error rate: 18.92%
Min DCF for Logistic Regression Weighted Quadratic: 0.32362704918032786

### lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5561475409836065

Logistic Regression Weighted Quadratic results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted Quadratic: 0.3643237704918033

### lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5642827868852459

Logistic Regression Weighted Quadratic results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted Quadratic: 0.39118852459016396

### lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5655327868852459

Logistic Regression Weighted Quadratic results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted Quadratic: 0.4027049180327869

## RAW + PCA with M = 8

### lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.46774590163934426

Logistic Regression Weighted Quadratic results:
Accuracy: 93.29%
Error rate: 6.71%
Min DCF for Logistic Regression Weighted Quadratic: 0.2983606557377049

### lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.46774590163934426

Logistic Regression Weighted Quadratic results:
Accuracy: 93.29%
Error rate: 6.71%
Min DCF for Logistic Regression Weighted Quadratic: 0.29866803278688525

### lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.46774590163934426

Logistic Regression Weighted Quadratic results:
Accuracy: 93.2%
Error rate: 6.8%
Min DCF for Logistic Regression Weighted Quadratic: 0.2958606557377049

### lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.46899590163934424

Logistic Regression Weighted Quadratic results:
Accuracy: 93.38%
Error rate: 6.62%
Min DCF for Logistic Regression Weighted Quadratic: 0.3008196721311476

### lambda value : 0.1 BEST LINEAR
Logistic Regression Weighted results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Logistic Regression Weighted: 0.4658606557377049

Logistic Regression Weighted Quadratic results:
Accuracy: 93.03%
Error rate: 6.97%
Min DCF for Logistic Regression Weighted Quadratic: 0.2973975409836066

### lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 77.85%
Error rate: 22.15%
Min DCF for Logistic Regression Weighted: 0.4849180327868853

Logistic Regression Weighted Quadratic results:
Accuracy: 92.69%
Error rate: 7.31%
Min DCF for Logistic Regression Weighted Quadratic: 0.28959016393442627

### lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5486680327868853

Logistic Regression Weighted Quadratic results:
Accuracy: 91.78%
Error rate: 8.22%
Min DCF for Logistic Regression Weighted Quadratic: 0.3155122950819672

### lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5791803278688525

Logistic Regression Weighted Quadratic results:
Accuracy: 88.3%
Error rate: 11.7%
Min DCF for Logistic Regression Weighted Quadratic: 0.3573565573770492

## Z_Norm + PCA with M = 8

### lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.14%
Error rate: 12.86%
Min DCF for Logistic Regression Weighted: 0.4833811475409836

Logistic Regression Weighted Quadratic results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for Logistic Regression Weighted Quadratic: 0.32737704918032784

### lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.4833811475409836

Logistic Regression Weighted Quadratic results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for Logistic Regression Weighted Quadratic: 0.32862704918032787

### lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.62%
Error rate: 13.38%
Min DCF for Logistic Regression Weighted: 0.4846311475409836

Logistic Regression Weighted Quadratic results:
Accuracy: 92.99%
Error rate: 7.01%
Min DCF for Logistic Regression Weighted Quadratic: 0.31737704918032783

### lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 83.91%
Error rate: 16.09%
Min DCF for Logistic Regression Weighted: 0.4968032786885246

Logistic Regression Weighted Quadratic results:
Accuracy: 91.48%
Error rate: 8.52%
Min DCF for Logistic Regression Weighted Quadratic: 0.30956967213114756

### lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.74%
Error rate: 32.26%
Min DCF for Logistic Regression Weighted: 0.5258196721311476

Logistic Regression Weighted Quadratic results:
Accuracy: 80.99%
Error rate: 19.01%
Min DCF for Logistic Regression Weighted Quadratic: 0.3302254098360656

### lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5608401639344263

Logistic Regression Weighted Quadratic results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted Quadratic: 0.37118852459016394

### lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5698975409836066

Logistic Regression Weighted Quadratic results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted Quadratic: 0.40334016393442623

### lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5723975409836065

Logistic Regression Weighted Quadratic results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted Quadratic: 0.4114549180327869

## Linear LR Best Values
### lambda 10^-1 PCA=8 no Znorm, try remaining combinations (0.1 and 0.9)
### Training Prior: Effective prior
#### DcfPrior: 0.1
Logistic Regression Weighted results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Logistic Regression Weighted: 0.7049999999999998

#### DcfPrior: 0.9
Logistic Regression Weighted results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Logistic Regression Weighted: 0.18325364298724953

### Training Prior: 0.5
#### DcfPrior: 0.5
Logistic Regression Weighted results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Logistic Regression Weighted: 0.47805327868852454

#### DcfPrior: 0.1
Logistic Regression Weighted results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Logistic Regression Weighted: 0.71625

#### DcfPrior: 0.9
Logistic Regression Weighted results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Logistic Regression Weighted: 0.18846083788706738

### Training Prior: 0.9
#### DcfPrior: 0.5
Logistic Regression Weighted results:
Accuracy: 78.84%
Error rate: 21.16%
Min DCF for Logistic Regression Weighted: 0.5102254098360656

#### DcfPrior: 0.1
Logistic Regression Weighted results:
Accuracy: 78.84%
Error rate: 21.16%
Min DCF for Logistic Regression Weighted: 0.71875

#### DcfPrior: 0.9
Logistic Regression Weighted results:
Accuracy: 78.84%
Error rate: 21.16%
Min DCF for Logistic Regression Weighted: 0.19428961748633877

### Training Prior: 0.1
#### DcfPrior: 0.5
Logistic Regression Weighted results:
Accuracy: 86.75%
Error rate: 13.25%
Min DCF for Logistic Regression Weighted: 0.4693032786885246

#### DcfPrior: 0.1
Logistic Regression Weighted results:
Accuracy: 86.75%
Error rate: 13.25%
Min DCF for Logistic Regression Weighted: 0.7049999999999998

#### DcfPrior: 0.9
Logistic Regression Weighted results:
Accuracy: 86.75%
Error rate: 13.25%
Min DCF for Logistic Regression Weighted: 0.18326047358834244

## Quadratic LR Best Values
### lambda = 10^-3 no PCA zNorm, try remaining combinations (0.1 and 0.9)

### Training Prior: Effective Prior
#### DcfPrior: 0.5
Logistic Regression Weighted Quadratic results:
Accuracy: 93.42%
Error rate: 6.58%
Min DCF for Logistic Regression Weighted Quadratic: 0.2852254098360656

#### DcfPrior: 0.1
Logistic Regression Weighted Quadratic results:
Accuracy: 93.42%
Error rate: 6.58%
Min DCF for Logistic Regression Weighted Quadratic: 0.5927663934426228

#### DcfPrior: 0.9
Logistic Regression Weighted Quadratic results:
Accuracy: 93.42%
Error rate: 6.58%
Min DCF for Logistic Regression Weighted Quadratic: 0.10390938069216757

### Training Prior: 0.5
#### DcfPrior: 0.5
Logistic Regression Weighted Quadratic results:
Accuracy: 95.05%
Error rate: 4.95%
Min DCF for Logistic Regression Weighted Quadratic: 0.3042827868852459

#### DcfPrior: 0.1
Logistic Regression Weighted Quadratic results:
Accuracy: 95.05%
Error rate: 4.95%
Min DCF for Logistic Regression Weighted Quadratic: 0.6130327868852459

#### DcfPrior: 0.9
Logistic Regression Weighted Quadratic results:
Accuracy: 95.05%
Error rate: 4.95%
Min DCF for Logistic Regression Weighted Quadratic: 0.10265938069216757

### Training Prior: 0.9
#### DcfPrior: 0.5
Logistic Regression Weighted Quadratic results:
Accuracy: 92.04%
Error rate: 7.96%
Min DCF for Logistic Regression Weighted Quadratic: 0.31362704918032785

#### DcfPrior: 0.1
Logistic Regression Weighted Quadratic results:
Accuracy: 92.04%
Error rate: 7.96%
Min DCF for Logistic Regression Weighted Quadratic: 0.6475

#### DcfPrior: 0.9
Logistic Regression Weighted Quadratic results:
Accuracy: 92.04%
Error rate: 7.96%
Min DCF for Logistic Regression Weighted Quadratic: 0.10380919854280508

### Training Prior: 0.1
#### DcfPrior: 0.5
Logistic Regression Weighted Quadratic results:
Accuracy: 93.68%
Error rate: 6.32%
Min DCF for Logistic Regression Weighted Quadratic: 0.28772540983606554

#### DcfPrior: 0.1
Logistic Regression Weighted Quadratic results:
Accuracy: 93.68%
Error rate: 6.32%
Min DCF for Logistic Regression Weighted Quadratic: 0.5940163934426229

#### DcfPrior: 0.9
Logistic Regression Weighted Quadratic results:
Accuracy: 93.68%
Error rate: 6.32%
Min DCF for Logistic Regression Weighted Quadratic: 0.10463797814207648

# SVM LINEAR HYPERPARAMETERS K AND C TRAINING: (Prior = 0.5)

## RAW (No PCA No Z_Norm)

### k value : 1
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 76.99%
Error rate: 23.01%
Min DCF for Linear SVM: 0.9560860655737704

C value : 0.001
Linear SVM results:
Accuracy: 79.57%
Error rate: 20.43%
Min DCF for Linear SVM: 0.9033811475409836

C value : 0.01
Linear SVM results:
Accuracy: 87.18%
Error rate: 12.82%
Min DCF for Linear SVM: 0.6505327868852459

C value : 0.1
Linear SVM results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.5033401639344262

C value : 1.0
Linear SVM results:
Accuracy: 90.75%
Error rate: 9.25%
Min DCF for Linear SVM: 0.47963114754098357

C value : 10.0
Linear SVM results:
Accuracy: 91.18%
Error rate: 8.82%
Min DCF for Linear SVM: 0.4743237704918033

C value : 100.0
Linear SVM results:
Accuracy: 90.06%
Error rate: 9.94%
Min DCF for Linear SVM: 0.5261475409836066

### k value : 10
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 89.81%
Error rate: 10.19%
Min DCF for Linear SVM: 0.5710860655737705

C value : 0.001
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4811885245901639

C value : 0.01
Linear SVM results:
Accuracy: 90.88%
Error rate: 9.12%
Min DCF for Linear SVM: 0.47493852459016395

C value : 0.1
Linear SVM results:
Accuracy: 91.01%
Error rate: 8.99%
Min DCF for Linear SVM: 0.4655737704918033

C value : 1.0
Linear SVM results:
Accuracy: 91.18%
Error rate: 8.82%
Min DCF for Linear SVM: 0.47963114754098357

C value : 10.0
Linear SVM results:
Accuracy: 91.18%
Error rate: 8.82%
Min DCF for Linear SVM: 0.4733811475409836

C value : 100.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.49930327868852453

## Z_Norm

### k value : 1
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 88.6%
Error rate: 11.4%
Min DCF for Linear SVM: 0.5677049180327869

C value : 0.001
Linear SVM results:
Accuracy: 88.65%
Error rate: 11.35%
Min DCF for Linear SVM: 0.5511475409836065

C value : 0.01
Linear SVM results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.5042622950819672

C value : 0.1
Linear SVM results:
Accuracy: 90.8%
Error rate: 9.2%
Min DCF for Linear SVM: 0.4943237704918033

C value : 1.0
Linear SVM results:
Accuracy: 90.84%
Error rate: 9.16%
Min DCF for Linear SVM: 0.483688524590164

C value : 10.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.4843237704918033

C value : 100.0
Linear SVM results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.4893237704918033

### k value : 10
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 0.5851434426229508

C value : 0.001
Linear SVM results:
Accuracy: 66.02%
Error rate: 33.98%
Min DCF for Linear SVM: 0.5454918032786885

C value : 0.01
Linear SVM results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.4993237704918033

C value : 0.1
Linear SVM results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Linear SVM: 0.4958811475409836

C value : 1.0
Linear SVM results:
Accuracy: 90.75%
Error rate: 9.25%
Min DCF for Linear SVM: 0.4824385245901639

C value : 10.0
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4883811475409836

C value : 100.0
Linear SVM results:
Accuracy: 89.72%
Error rate: 10.28%
Min DCF for Linear SVM: 0.5058401639344262

## RAW + PCA with M = 8

### k value : 1
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 76.86%
Error rate: 23.14%
Min DCF for Linear SVM: 0.9487295081967213

C value : 0.001
Linear SVM results:
Accuracy: 79.78%
Error rate: 20.22%
Min DCF for Linear SVM: 0.9108401639344262

C value : 0.01
Linear SVM results:
Accuracy: 86.58%
Error rate: 13.42%
Min DCF for Linear SVM: 0.6457786885245902

C value : 0.1
Linear SVM results:
Accuracy: 90.75%
Error rate: 9.25%
Min DCF for Linear SVM: 0.48428278688524595

C value : 1.0
Linear SVM results:
Accuracy: 91.01%
Error rate: 8.99%
Min DCF for Linear SVM: 0.4918032786885246

C value : 10.0
Linear SVM results:
Accuracy: 90.97%
Error rate: 9.03%
Min DCF for Linear SVM: 0.4839754098360656

C value : 100.0
Linear SVM results:
Accuracy: 88.13%
Error rate: 11.87%
Min DCF for Linear SVM: 0.5980122950819672

### k value : 10
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 89.33%
Error rate: 10.67%
Min DCF for Linear SVM: 0.5695081967213115

C value : 0.001
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.4896311475409836

C value : 0.01
Linear SVM results:
Accuracy: 91.01%
Error rate: 8.99%
Min DCF for Linear SVM: 0.4833606557377049

C value : 0.1
Linear SVM results:
Accuracy: 90.97%
Error rate: 9.03%
Min DCF for Linear SVM: 0.4889754098360656

C value : 1.0
Linear SVM results:
Accuracy: 90.97%
Error rate: 9.03%
Min DCF for Linear SVM: 0.4827254098360656

C value : 10.0
Linear SVM results:
Accuracy: 91.01%
Error rate: 8.99%
Min DCF for Linear SVM: 0.4827254098360656

C value : 100.0
Linear SVM results:
Accuracy: 89.42%
Error rate: 10.58%
Min DCF for Linear SVM: 0.5817622950819672

## Z_Norm + PCA with M = 8

### k value : 1
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 88.65%
Error rate: 11.35%
Min DCF for Linear SVM: 0.5682991803278689

C value : 0.001
Linear SVM results:
Accuracy: 88.69%
Error rate: 11.31%
Min DCF for Linear SVM: 0.5539754098360656

C value : 0.01
Linear SVM results:
Accuracy: 90.37%
Error rate: 9.63%
Min DCF for Linear SVM: 0.502766393442623

C value : 0.1
Linear SVM results:
Accuracy: 90.88%
Error rate: 9.12%
Min DCF for Linear SVM: 0.48618852459016393

C value : 1.0
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4996311475409836

C value : 10.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.5074385245901639

C value : 100.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.5058811475409837

### k value : 10
C value : 1e-06
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 1e-05
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 1.0

C value : 0.0001
Linear SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Linear SVM: 0.6032581967213115

C value : 0.001
Linear SVM results:
Accuracy: 66.02%
Error rate: 33.98%
Min DCF for Linear SVM: 0.5439139344262296

C value : 0.01
Linear SVM results:
Accuracy: 90.92%
Error rate: 9.08%
Min DCF for Linear SVM: 0.5049590163934427

C value : 0.1
Linear SVM results:
Accuracy: 90.92%
Error rate: 9.08%
Min DCF for Linear SVM: 0.49493852459016396

C value : 1.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.5024385245901639

C value : 10.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.5071311475409837

C value : 100.0
Linear SVM results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Linear SVM: 0.5055327868852459

# SVM POLYNOMIAL K,C,c,d TRAINING: (Prior = 0.5)

## RAW (No PCA No Z_Norm)

k value : 1
C value : 1e-06
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 1e-05
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 0.0001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.62%
Error rate: 5.38%
Min DCF for Polynomial Kernel SVM: 0.31989754098360657

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.67%
Error rate: 5.33%
Min DCF for Polynomial Kernel SVM: 0.3083401639344262

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Polynomial Kernel SVM: 0.3308401639344262

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.75%
Error rate: 5.25%
Min DCF for Polynomial Kernel SVM: 0.31114754098360653

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.41%
Error rate: 5.59%
Min DCF for Polynomial Kernel SVM: 0.3427049180327869

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.67%
Error rate: 5.33%
Min DCF for Polynomial Kernel SVM: 0.31456967213114756

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.92%
Error rate: 5.08%
Min DCF for Polynomial Kernel SVM: 0.34424180327868853

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.67%
Error rate: 5.33%
Min DCF for Polynomial Kernel SVM: 0.30959016393442623

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for Polynomial Kernel SVM: 0.3579918032786885

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.58%
Error rate: 5.42%
Min DCF for Polynomial Kernel SVM: 0.3392418032786886

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 86.06%
Error rate: 13.94%
Min DCF for Polynomial Kernel SVM: 0.6167213114754099

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 79.1%
Error rate: 20.9%
Min DCF for Polynomial Kernel SVM: 0.9774180327868853

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 63.7%
Error rate: 36.3%
Min DCF for Polynomial Kernel SVM: 0.9853073770491804

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 58.71%
Error rate: 41.29%
Min DCF for Polynomial Kernel SVM: 1.0

## Z_Norm

k value : 1
C value : 1e-06
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 1e-05
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 0.0001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.63%
Error rate: 34.37%
Min DCF for Polynomial Kernel SVM: 0.8241393442622951

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 82.54%
Error rate: 17.46%
Min DCF for Polynomial Kernel SVM: 0.40864754098360656

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 82.28%
Error rate: 17.72%
Min DCF for Polynomial Kernel SVM: 0.7188729508196722

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.71%
Error rate: 5.29%
Min DCF for Polynomial Kernel SVM: 0.31797131147540986

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.91%
Error rate: 12.09%
Min DCF for Polynomial Kernel SVM: 0.6800819672131148

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.48%
Error rate: 4.52%
Min DCF for Polynomial Kernel SVM: 0.30116803278688525

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 89.25%
Error rate: 10.75%
Min DCF for Polynomial Kernel SVM: 0.6286270491803279

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.27%
Error rate: 4.73%
Min DCF for Polynomial Kernel SVM: 0.3023975409836066

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 88.99%
Error rate: 11.01%
Min DCF for Polynomial Kernel SVM: 0.6479303278688524

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.92%
Error rate: 5.08%
Min DCF for Polynomial Kernel SVM: 0.3176844262295082

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 89.12%
Error rate: 10.88%
Min DCF for Polynomial Kernel SVM: 0.6541803278688525

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.54%
Error rate: 5.46%
Min DCF for Polynomial Kernel SVM: 0.3345696721311475

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 89.03%
Error rate: 10.97%
Min DCF for Polynomial Kernel SVM: 0.6566803278688524

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Polynomial Kernel SVM: 0.33581967213114755

## RAW + PCA with M = 8

k value : 1
C value : 1e-06
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 1e-05
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 0.0001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.37%
Error rate: 5.63%
Min DCF for Polynomial Kernel SVM: 0.32178278688524586

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.58%
Error rate: 5.42%
Min DCF for Polynomial Kernel SVM: 0.3108196721311476

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.41%
Error rate: 5.59%
Min DCF for Polynomial Kernel SVM: 0.3096106557377049

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.62%
Error rate: 5.38%
Min DCF for Polynomial Kernel SVM: 0.3067827868852459

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for Polynomial Kernel SVM: 0.31772540983606556

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.54%
Error rate: 5.46%
Min DCF for Polynomial Kernel SVM: 0.3217418032786885

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Polynomial Kernel SVM: 0.32641393442622957

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.1%
Error rate: 4.9%
Min DCF for Polynomial Kernel SVM: 0.3120491803278689

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.58%
Error rate: 5.42%
Min DCF for Polynomial Kernel SVM: 0.3386680327868853

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.11%
Error rate: 5.89%
Min DCF for Polynomial Kernel SVM: 0.35362704918032783

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 84.99%
Error rate: 15.01%
Min DCF for Polynomial Kernel SVM: 0.6598975409836066

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 86.19%
Error rate: 13.81%
Min DCF for Polynomial Kernel SVM: 0.9538729508196722

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 68.43%
Error rate: 31.57%
Min DCF for Polynomial Kernel SVM: 0.9931147540983606

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 62.02%
Error rate: 37.98%
Min DCF for Polynomial Kernel SVM: 0.99375

## Z_Norm + PCA with M = 8

k value : 1
C value : 1e-06
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 1e-05
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 0.0001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.63%
Error rate: 34.37%
Min DCF for Polynomial Kernel SVM: 0.8469877049180328

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 82.92%
Error rate: 17.08%
Min DCF for Polynomial Kernel SVM: 0.4076844262295082

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 82.02%
Error rate: 17.98%
Min DCF for Polynomial Kernel SVM: 0.7288729508196722

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.62%
Error rate: 5.38%
Min DCF for Polynomial Kernel SVM: 0.3192213114754099

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.18%
Error rate: 12.82%
Min DCF for Polynomial Kernel SVM: 0.7238524590163935

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.31%
Error rate: 4.69%
Min DCF for Polynomial Kernel SVM: 0.29711065573770495

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 88.17%
Error rate: 11.83%
Min DCF for Polynomial Kernel SVM: 0.6713934426229509

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.57%
Error rate: 4.43%
Min DCF for Polynomial Kernel SVM: 0.2986475409836066

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.96%
Error rate: 12.04%
Min DCF for Polynomial Kernel SVM: 0.6370081967213115

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.27%
Error rate: 4.73%
Min DCF for Polynomial Kernel SVM: 0.3308401639344262

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.87%
Error rate: 12.13%
Min DCF for Polynomial Kernel SVM: 0.6479303278688524

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.35%
Error rate: 4.65%
Min DCF for Polynomial Kernel SVM: 0.3264549180327869

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.91%
Error rate: 12.09%
Min DCF for Polynomial Kernel SVM: 0.6516803278688525

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.35%
Error rate: 4.65%
Min DCF for Polynomial Kernel SVM: 0.3252049180327869