# Models

- [Gaussians](#gaussians)
- [Logistic Regression](#lr-prior--05)
    - [Best Values Linear](#linear-lr-best-values)
    - [Best Values Quadratic](#quadratic-lr-best-values)
- [Linear SVM](#svm-linear-hyperparameters-k-and-c-training-prior--05)
- [Polynomial SVM](#svm-polynomial-kccd-training-prior--05)
- [RBF SVM](#svm-radial-basis-function-rbf-kcgamma-training-prior--05)
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
Accuracy: 79.01%
Error rate: 20.99%
Min DCF for Linear SVM: 0.9230737704918033

C value : 0.001
Linear SVM results:
Accuracy: 82.97%
Error rate: 17.03%
Min DCF for Linear SVM: 0.7748155737704918

C value : 0.01
Linear SVM results:
Accuracy: 89.81%
Error rate: 10.19%
Min DCF for Linear SVM: 0.5320286885245902

C value : 0.1
Linear SVM results:
Accuracy: 90.84%
Error rate: 9.16%
Min DCF for Linear SVM: 0.47649590163934424

C value : 1.0
Linear SVM results:
Accuracy: 91.01%
Error rate: 8.99%
Min DCF for Linear SVM: 0.4680737704918033

C value : 10.0
Linear SVM results:
Accuracy: 89.94%
Error rate: 10.06%
Min DCF for Linear SVM: 0.5420901639344262

C value : 100.0
Linear SVM results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Linear SVM: 0.7947950819672132

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
Accuracy: 91.05%
Error rate: 8.95%
Min DCF for Linear SVM: 0.49118852459016393

C value : 0.001
Linear SVM results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Linear SVM: 0.4830737704918033

C value : 0.01
Linear SVM results:
Accuracy: 91.1%
Error rate: 8.9%
Min DCF for Linear SVM: 0.47307377049180327

C value : 0.1
Linear SVM results:
Accuracy: 91.14%
Error rate: 8.86%
Min DCF for Linear SVM: 0.4793032786885246

C value : 1.0
Linear SVM results:
Accuracy: 91.14%
Error rate: 8.86%
Min DCF for Linear SVM: 0.47776639344262295

C value : 10.0
Linear SVM results:
Accuracy: 90.97%
Error rate: 9.03%
Min DCF for Linear SVM: 0.47491803278688527

C value : 100.0
Linear SVM results:
Accuracy: 82.41%
Error rate: 17.59%
Min DCF for Linear SVM: 0.667438524590164

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
Min DCF for Linear SVM: 0.5655327868852459

C value : 0.001
Linear SVM results:
Accuracy: 90.06%
Error rate: 9.94%
Min DCF for Linear SVM: 0.512704918032787

C value : 0.01
Linear SVM results:
Accuracy: 90.92%
Error rate: 9.08%
Min DCF for Linear SVM: 0.48651639344262293

C value : 0.1
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4858811475409836

C value : 1.0
Linear SVM results:
Accuracy: 90.8%
Error rate: 9.2%
Min DCF for Linear SVM: 0.4871311475409836

C value : 10.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.4905737704918033

C value : 100.0
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4905737704918033

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
Min DCF for Linear SVM: 0.5676434426229509

C value : 0.001
Linear SVM results:
Accuracy: 90.11%
Error rate: 9.89%
Min DCF for Linear SVM: 0.5186270491803279

C value : 0.01
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4955532786885246

C value : 0.1
Linear SVM results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.49149590163934426

C value : 1.0
Linear SVM results:
Accuracy: 90.75%
Error rate: 9.25%
Min DCF for Linear SVM: 0.4868237704918033

C value : 10.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.4924385245901639

C value : 100.0
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.4821311475409836

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
Accuracy: 78.97%
Error rate: 21.03%
Min DCF for Linear SVM: 0.9318237704918032

C value : 0.001
Linear SVM results:
Accuracy: 82.58%
Error rate: 17.42%
Min DCF for Linear SVM: 0.7829713114754099

C value : 0.01
Linear SVM results:
Accuracy: 89.76%
Error rate: 10.24%
Min DCF for Linear SVM: 0.5254508196721311

C value : 0.1
Linear SVM results:
Accuracy: 90.8%
Error rate: 9.2%
Min DCF for Linear SVM: 0.4833606557377049

C value : 1.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.4974385245901639

C value : 10.0
Linear SVM results:
Accuracy: 91.1%
Error rate: 8.9%
Min DCF for Linear SVM: 0.47805327868852454

C value : 100.0
Linear SVM results:
Accuracy: 74.84%
Error rate: 25.16%
Min DCF for Linear SVM: 0.9975

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
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.4936885245901639

C value : 0.001
Linear SVM results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Linear SVM: 0.47899590163934425

C value : 0.01
Linear SVM results:
Accuracy: 90.88%
Error rate: 9.12%
Min DCF for Linear SVM: 0.48399590163934425

C value : 0.1
Linear SVM results:
Accuracy: 91.01%
Error rate: 8.99%
Min DCF for Linear SVM: 0.4852254098360656

C value : 1.0
Linear SVM results:
Accuracy: 90.97%
Error rate: 9.03%
Min DCF for Linear SVM: 0.4827254098360656

C value : 10.0
Linear SVM results:
Accuracy: 90.88%
Error rate: 9.12%
Min DCF for Linear SVM: 0.49649590163934426

C value : 100.0
Linear SVM results:
Accuracy: 80.26%
Error rate: 19.74%
Min DCF for Linear SVM: 0.98375

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
Min DCF for Linear SVM: 0.5736475409836066

C value : 0.001
Linear SVM results:
Accuracy: 90.15%
Error rate: 9.85%
Min DCF for Linear SVM: 0.5182991803278688

C value : 0.01
Linear SVM results:
Accuracy: 90.84%
Error rate: 9.16%
Min DCF for Linear SVM: 0.4905737704918033

C value : 0.1
Linear SVM results:
Accuracy: 90.62%
Error rate: 9.38%
Min DCF for Linear SVM: 0.49401639344262294

C value : 1.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.5074385245901639

C value : 10.0
Linear SVM results:
Accuracy: 90.67%
Error rate: 9.33%
Min DCF for Linear SVM: 0.5058811475409837

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
Min DCF for Linear SVM: 0.5654713114754099

C value : 0.001
Linear SVM results:
Accuracy: 89.94%
Error rate: 10.06%
Min DCF for Linear SVM: 0.5139344262295082

C value : 0.01
Linear SVM results:
Accuracy: 90.97%
Error rate: 9.03%
Min DCF for Linear SVM: 0.49149590163934426

C value : 0.1
Linear SVM results:
Accuracy: 90.58%
Error rate: 9.42%
Min DCF for Linear SVM: 0.4933811475409836

C value : 1.0
Linear SVM results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Linear SVM: 0.5046311475409836

C value : 10.0
Linear SVM results:
Accuracy: 90.71%
Error rate: 9.29%
Min DCF for Linear SVM: 0.5145901639344262

C value : 100.0
Linear SVM results:
Accuracy: 90.11%
Error rate: 9.89%
Min DCF for Linear SVM: 0.48834016393442625

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
Accuracy: 94.54%
Error rate: 5.46%
Min DCF for Polynomial Kernel SVM: 0.3230327868852459

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.67%
Error rate: 5.33%
Min DCF for Polynomial Kernel SVM: 0.31426229508196724

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.37%
Error rate: 5.63%
Min DCF for Polynomial Kernel SVM: 0.3373975409836065

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.54%
Error rate: 5.46%
Min DCF for Polynomial Kernel SVM: 0.3023975409836066

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.67%
Error rate: 5.33%
Min DCF for Polynomial Kernel SVM: 0.34551229508196724

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.75%
Error rate: 5.25%
Min DCF for Polynomial Kernel SVM: 0.30959016393442623

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Polynomial Kernel SVM: 0.33584016393442623

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.23%
Error rate: 4.77%
Min DCF for Polynomial Kernel SVM: 0.31831967213114754

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.06%
Error rate: 5.94%
Min DCF for Polynomial Kernel SVM: 0.35834016393442625

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 92.26%
Error rate: 7.74%
Min DCF for Polynomial Kernel SVM: 0.42991803278688523

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 70.58%
Error rate: 29.42%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 72.39%
Error rate: 27.61%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 65.51%
Error rate: 34.49%
Min DCF for Polynomial Kernel SVM: 0.9978073770491803

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 68.13%
Error rate: 31.87%
Min DCF for Polynomial Kernel SVM: 0.9925

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
Accuracy: 72.17%
Error rate: 27.83%
Min DCF for Polynomial Kernel SVM: 0.6801024590163934

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 92.47%
Error rate: 7.53%
Min DCF for Polynomial Kernel SVM: 0.3333811475409836

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.48%
Error rate: 12.52%
Min DCF for Polynomial Kernel SVM: 0.6878893442622951

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.31%
Error rate: 4.69%
Min DCF for Polynomial Kernel SVM: 0.29270491803278686

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 88.56%
Error rate: 11.44%
Min DCF for Polynomial Kernel SVM: 0.6320286885245902

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.27%
Error rate: 4.73%
Min DCF for Polynomial Kernel SVM: 0.2995491803278688

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 89.08%
Error rate: 10.92%
Min DCF for Polynomial Kernel SVM: 0.6410245901639344

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.92%
Error rate: 5.08%
Min DCF for Polynomial Kernel SVM: 0.30926229508196723

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 89.2%
Error rate: 10.8%
Min DCF for Polynomial Kernel SVM: 0.6535450819672131

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.58%
Error rate: 5.42%
Min DCF for Polynomial Kernel SVM: 0.3267622950819672

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 88.99%
Error rate: 11.01%
Min DCF for Polynomial Kernel SVM: 0.6594672131147541

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Polynomial Kernel SVM: 0.33831967213114755

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 89.16%
Error rate: 10.84%
Min DCF for Polynomial Kernel SVM: 0.6385450819672132

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Polynomial Kernel SVM: 0.3330122950819672

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
Accuracy: 94.45%
Error rate: 5.55%
Min DCF for Polynomial Kernel SVM: 0.32397540983606554

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.62%
Error rate: 5.38%
Min DCF for Polynomial Kernel SVM: 0.30459016393442623

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.32%
Error rate: 5.68%
Min DCF for Polynomial Kernel SVM: 0.32022540983606557

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.54%
Error rate: 5.46%
Min DCF for Polynomial Kernel SVM: 0.30801229508196726

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for Polynomial Kernel SVM: 0.32897540983606555

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.92%
Error rate: 5.08%
Min DCF for Polynomial Kernel SVM: 0.3198565573770492

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 94.84%
Error rate: 5.16%
Min DCF for Polynomial Kernel SVM: 0.33487704918032785

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.01%
Error rate: 4.99%
Min DCF for Polynomial Kernel SVM: 0.2930532786885246

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 85.89%
Error rate: 14.11%
Min DCF for Polynomial Kernel SVM: 0.6217622950819672

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 92.26%
Error rate: 7.74%
Min DCF for Polynomial Kernel SVM: 0.4677049180327869

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 62.02%
Error rate: 37.98%
Min DCF for Polynomial Kernel SVM: 1.0

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 68.9%
Error rate: 31.1%
Min DCF for Polynomial Kernel SVM: 1.0

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 60.34%
Error rate: 39.66%
Min DCF for Polynomial Kernel SVM: 0.995

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 59.31%
Error rate: 40.69%
Min DCF for Polynomial Kernel SVM: 1.0

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
Accuracy: 72.22%
Error rate: 27.78%
Min DCF for Polynomial Kernel SVM: 0.7051229508196721

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 92.17%
Error rate: 7.83%
Min DCF for Polynomial Kernel SVM: 0.3433606557377049

C value : 0.001
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 86.71%
Error rate: 13.29%
Min DCF for Polynomial Kernel SVM: 0.7432172131147541

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.27%
Error rate: 4.73%
Min DCF for Polynomial Kernel SVM: 0.31112704918032785

C value : 0.01
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 88.09%
Error rate: 11.91%
Min DCF for Polynomial Kernel SVM: 0.6995286885245902

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.31%
Error rate: 4.69%
Min DCF for Polynomial Kernel SVM: 0.28397540983606556

C value : 0.1
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 88.13%
Error rate: 11.87%
Min DCF for Polynomial Kernel SVM: 0.6429303278688525

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.23%
Error rate: 4.77%
Min DCF for Polynomial Kernel SVM: 0.32178278688524586

C value : 1.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.96%
Error rate: 12.04%
Min DCF for Polynomial Kernel SVM: 0.6479303278688524

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.35%
Error rate: 4.65%
Min DCF for Polynomial Kernel SVM: 0.3186475409836066

C value : 10.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.96%
Error rate: 12.04%
Min DCF for Polynomial Kernel SVM: 0.6491803278688525

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.35%
Error rate: 4.65%
Min DCF for Polynomial Kernel SVM: 0.3252049180327869

C value : 100.0
c value : 0
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 87.96%
Error rate: 12.04%
Min DCF for Polynomial Kernel SVM: 0.6523155737704918

c value : 1
d value : 2.0
Polynomial Kernel SVM results:
Accuracy: 95.31%
Error rate: 4.69%
Min DCF for Polynomial Kernel SVM: 0.3308401639344262

# SVM RADIAL BASIS FUNCTION (RBF) K,C,gamma TRAINING: (Prior = 0.5)

## RAW (No PCA No Z_Norm)

k value : 1.0
C value : 1e-06
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 1e-05
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 0.0001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9159016393442623

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.8984016393442623

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.6430122950819672

C value : 0.001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9159016393442623

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.8984016393442623

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.6430122950819672

C value : 0.01
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5279303278688525

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.46454918032786885

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 71.48%
Error rate: 28.52%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42793032786885243

C value : 0.1
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5451024590163934

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 76.6%
Error rate: 23.4%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4564344262295082

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.02%
Error rate: 5.98%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3585860655737705

C value : 1.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 71.48%
Error rate: 28.52%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5513524590163934

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.83%
Error rate: 8.17%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.43983606557377053

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.71%
Error rate: 5.29%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3267418032786885

C value : 10.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 74.37%
Error rate: 25.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5563729508196721

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.91%
Error rate: 8.09%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4401229508196721

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.19%
Error rate: 5.81%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.39454918032786884

C value : 100.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 74.37%
Error rate: 25.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5563729508196721

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.91%
Error rate: 8.09%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4401229508196721

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42981557377049184

C value : 1000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 74.37%
Error rate: 25.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5563729508196721

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.91%
Error rate: 8.09%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4401229508196721

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42981557377049184

C value : 10000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 74.37%
Error rate: 25.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5563729508196721

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.91%
Error rate: 8.09%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4401229508196721

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42981557377049184

C value : 100000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 74.37%
Error rate: 25.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5563729508196721

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.91%
Error rate: 8.09%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4401229508196721

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42981557377049184

## Z_Norm

k value : 1.0
C value : 1e-06
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 1e-05
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 0.0001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.7667827868852459

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9631147540983606

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.98375

C value : 0.001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.7667827868852459

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9202459016393443

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9652459016393443

C value : 0.01
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 88.77%
Error rate: 11.23%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.39737704918032785

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 66.54%
Error rate: 33.46%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.47993852459016395

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5417008196721311

C value : 0.1
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.54%
Error rate: 5.46%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.34422131147540985

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.72%
Error rate: 6.28%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3989139344262295

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.57%
Error rate: 8.43%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.46709016393442626

C value : 1.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.1%
Error rate: 4.9%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31895491803278686

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.32%
Error rate: 5.68%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3255122950819672

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 92.95%
Error rate: 7.05%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42834016393442625

C value : 10.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.84%
Error rate: 5.16%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31956967213114756

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.18%
Error rate: 4.82%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.30959016393442623

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.32%
Error rate: 5.68%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.33954918032786885

C value : 100.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.58%
Error rate: 5.42%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.35797131147540984

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.8%
Error rate: 5.2%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31209016393442623

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.23%
Error rate: 4.77%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3105327868852459

C value : 1000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 92.65%
Error rate: 7.35%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.46575819672131147

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.62%
Error rate: 5.38%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.34581967213114756

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.88%
Error rate: 5.12%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31643442622950824

C value : 10000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.53%
Error rate: 8.47%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5394877049180328

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.12%
Error rate: 6.88%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4447950819672131

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.62%
Error rate: 5.38%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3623565573770492

C value : 100000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.57%
Error rate: 8.43%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5473360655737705

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.7%
Error rate: 8.3%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5510040983606557

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 84.73%
Error rate: 15.27%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.8494877049180327

## RAW + PCA with M = 8

k value : 1.0
C value : 1e-06
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 1e-05
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 0.0001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9015368852459017

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.8859016393442622

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5095491803278689

C value : 0.001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9015368852459017

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.8859016393442622

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5095491803278689

C value : 0.01
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4770491803278688

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4401434426229508

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 83.7%
Error rate: 16.3%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.40827868852459015

C value : 0.1
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4917213114754099

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 86.37%
Error rate: 13.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.41077868852459015

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.58%
Error rate: 5.42%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.32612704918032787

C value : 1.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 83.78%
Error rate: 16.22%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.49641393442622955

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.72%
Error rate: 6.28%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3976639344262295

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.23%
Error rate: 4.77%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31579918032786886

C value : 10.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 85.08%
Error rate: 14.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.49456967213114755

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.29%
Error rate: 6.71%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4049180327868852

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.94%
Error rate: 6.06%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3720491803278688

C value : 100.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 85.08%
Error rate: 14.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.49456967213114755

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.08%
Error rate: 6.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4192827868852459

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.16%
Error rate: 6.84%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.43983606557377053

C value : 1000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 85.08%
Error rate: 14.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.49456967213114755

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.08%
Error rate: 6.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4192827868852459

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 92.22%
Error rate: 7.78%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4120901639344262

C value : 10000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 85.08%
Error rate: 14.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.49456967213114755

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.08%
Error rate: 6.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4192827868852459

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 92.22%
Error rate: 7.78%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4120901639344262

C value : 100000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 85.08%
Error rate: 14.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.49456967213114755

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.08%
Error rate: 6.92%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4192827868852459

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 92.22%
Error rate: 7.78%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4120901639344262

## Z_Norm + PCA with M = 8

k value : 1.0
C value : 1e-06
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 1e-05
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 1.0

C value : 0.0001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.7542827868852459

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.96

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9806147540983606

C value : 0.001
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.7542827868852459

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.9030532786885246

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.951188524590164

C value : 0.01
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 90.54%
Error rate: 9.46%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.39827868852459014

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 66.84%
Error rate: 33.16%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.48149590163934425

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5542008196721311

C value : 0.1
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.92%
Error rate: 5.08%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.34331967213114756

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.63%
Error rate: 6.37%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.39170081967213116

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.57%
Error rate: 8.43%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.46993852459016394

C value : 1.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.05%
Error rate: 4.95%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3196106557377049

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.37%
Error rate: 5.63%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.33331967213114755

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.33%
Error rate: 6.67%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4251639344262295

C value : 10.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.01%
Error rate: 4.99%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31084016393442626

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.23%
Error rate: 4.77%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.30709016393442623

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.49%
Error rate: 5.51%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.32581967213114754

C value : 100.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.24%
Error rate: 5.76%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.35079918032786883

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.01%
Error rate: 4.99%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.31395491803278686

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.44%
Error rate: 4.56%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.30301229508196725

C value : 1000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.16%
Error rate: 6.84%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4429508196721311

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 94.67%
Error rate: 5.33%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.34424180327868853

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 95.14%
Error rate: 4.86%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.3236475409836066

C value : 10000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 91.53%
Error rate: 8.47%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.4963934426229508

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 92.73%
Error rate: 7.27%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.42670081967213114

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 93.94%
Error rate: 6.06%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.37086065573770494

C value : 100000.0
gamma value : 0.049787068367863944
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 90.88%
Error rate: 9.12%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.5235655737704918

gamma value : 0.018315638888734182
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 81.85%
Error rate: 18.15%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.7976639344262295

gamma value : 0.006737946999085467
RADIAL BASIS FUNCTION (RBF) Kernel SVM results:
Accuracy: 73.16%
Error rate: 26.84%
Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: 0.99