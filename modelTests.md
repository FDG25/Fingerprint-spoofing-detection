# Models

- [Gaussians](#gaussians)
- [Linear Logistic Regression](#linear-lr)

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

# Linear LR 

## Prior = 0.5

### RAW (No PCA No Z_Norm)

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.47116803278688524

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.47116803278688524

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.47116803278688524

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.23%
Error rate: 12.77%
Min DCF for Logistic Regression Weighted: 0.4736680327868853

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.41%
Error rate: 13.59%
Min DCF for Logistic Regression Weighted: 0.4771311475409836

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 78.19%
Error rate: 21.81%
Min DCF for Logistic Regression Weighted: 0.4889549180327869

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.553360655737705

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5823770491803278

### Z_Norm

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.05%
Error rate: 12.95%
Min DCF for Logistic Regression Weighted: 0.4852459016393443

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 86.97%
Error rate: 13.03%
Min DCF for Logistic Regression Weighted: 0.4852459016393443

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.49%
Error rate: 13.51%
Min DCF for Logistic Regression Weighted: 0.4868032786885246

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 84.09%
Error rate: 15.91%
Min DCF for Logistic Regression Weighted: 0.4974385245901639

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.83%
Error rate: 32.17%
Min DCF for Logistic Regression Weighted: 0.5177049180327868

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5561475409836065

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5642827868852459

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5655327868852459

### RAW + PCA with M = 8

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.46774590163934426

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.46774590163934426

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.46774590163934426

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.46899590163934424

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Logistic Regression Weighted: 0.4658606557377049

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 77.85%
Error rate: 22.15%
Min DCF for Logistic Regression Weighted: 0.4849180327868853

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5486680327868853

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5791803278688525

### Z_Norm + PCA with M = 8

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.14%
Error rate: 12.86%
Min DCF for Logistic Regression Weighted: 0.4833811475409836

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.4833811475409836

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.62%
Error rate: 13.38%
Min DCF for Logistic Regression Weighted: 0.4846311475409836

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 83.91%
Error rate: 16.09%
Min DCF for Logistic Regression Weighted: 0.4968032786885246

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.74%
Error rate: 32.26%
Min DCF for Logistic Regression Weighted: 0.5258196721311476

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5608401639344263

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5698975409836066

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.5723975409836065

## Prior = 0.1

### RAW (No PCA No Z_Norm)

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.720266393442623

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.720266393442623

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.720266393442623

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.23%
Error rate: 12.77%
Min DCF for Logistic Regression Weighted: 0.721516393442623

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.41%
Error rate: 13.59%
Min DCF for Logistic Regression Weighted: 0.72125

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 78.19%
Error rate: 21.81%
Min DCF for Logistic Regression Weighted: 0.70375

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.73625

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.75375

### Z_Norm

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.05%
Error rate: 12.95%
Min DCF for Logistic Regression Weighted: 0.71125

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 86.97%
Error rate: 13.03%
Min DCF for Logistic Regression Weighted: 0.71125

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.49%
Error rate: 13.51%
Min DCF for Logistic Regression Weighted: 0.7075

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 84.09%
Error rate: 15.91%
Min DCF for Logistic Regression Weighted: 0.7049999999999998

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.83%
Error rate: 32.17%
Min DCF for Logistic Regression Weighted: 0.714016393442623

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.732766393442623

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.752766393442623

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.752766393442623

### RAW + PCA with M = 8

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.701516393442623

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.701516393442623

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.701516393442623

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.701516393442623

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Logistic Regression Weighted: 0.7049999999999998

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 77.85%
Error rate: 22.15%
Min DCF for Logistic Regression Weighted: 0.68875

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.72625

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.75875

### Z_Norm + PCA with M = 8

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.14%
Error rate: 12.86%
Min DCF for Logistic Regression Weighted: 0.7107991803278687

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.7107991803278687

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.62%
Error rate: 13.38%
Min DCF for Logistic Regression Weighted: 0.7095491803278688

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 83.91%
Error rate: 16.09%
Min DCF for Logistic Regression Weighted: 0.7145491803278688

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.74%
Error rate: 32.26%
Min DCF for Logistic Regression Weighted: 0.715266393442623

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.729016393442623

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.752766393442623

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.755266393442623

## Prior = 0.9

### RAW (No PCA No Z_Norm)

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.1816962659380692

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.1816962659380692

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.53%
Error rate: 12.47%
Min DCF for Logistic Regression Weighted: 0.18117486338797814

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.23%
Error rate: 12.77%
Min DCF for Logistic Regression Weighted: 0.18117486338797814

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.41%
Error rate: 13.59%
Min DCF for Logistic Regression Weighted: 0.18241803278688523

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 78.19%
Error rate: 21.81%
Min DCF for Logistic Regression Weighted: 0.18431010928961747

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.2004394353369763

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.2103324225865209

### Z_Norm

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.05%
Error rate: 12.95%
Min DCF for Logistic Regression Weighted: 0.18158925318761385

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 86.97%
Error rate: 13.03%
Min DCF for Logistic Regression Weighted: 0.18158925318761385

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.49%
Error rate: 13.51%
Min DCF for Logistic Regression Weighted: 0.182103825136612

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 84.09%
Error rate: 15.91%
Min DCF for Logistic Regression Weighted: 0.18688296903460835

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.83%
Error rate: 32.17%
Min DCF for Logistic Regression Weighted: 0.19638205828779598

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.20273224043715846

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.20523224043715846

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.20491803278688522

### RAW + PCA with M = 8

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.18325364298724953

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.18325364298724953

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 87.01%
Error rate: 12.99%
Min DCF for Logistic Regression Weighted: 0.18325364298724953

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.18231785063752276

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 86.32%
Error rate: 13.68%
Min DCF for Logistic Regression Weighted: 0.18325364298724953

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 77.85%
Error rate: 22.15%
Min DCF for Logistic Regression Weighted: 0.1900250455373406

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.2022108378870674

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.20887522768670302

### Z_Norm + PCA with M = 8

lambda value : 1e-05
Logistic Regression Weighted results:
Accuracy: 87.14%
Error rate: 12.86%
Min DCF for Logistic Regression Weighted: 0.1886680327868852

lambda value : 0.0001
Logistic Regression Weighted results:
Accuracy: 87.1%
Error rate: 12.9%
Min DCF for Logistic Regression Weighted: 0.1886680327868852

lambda value : 0.001
Logistic Regression Weighted results:
Accuracy: 86.62%
Error rate: 13.38%
Min DCF for Logistic Regression Weighted: 0.189075591985428

lambda value : 0.01
Logistic Regression Weighted results:
Accuracy: 83.91%
Error rate: 16.09%
Min DCF for Logistic Regression Weighted: 0.18908242258652094

lambda value : 0.1
Logistic Regression Weighted results:
Accuracy: 67.74%
Error rate: 32.26%
Min DCF for Logistic Regression Weighted: 0.1982604735883424

lambda value : 1.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.2050250455373406

lambda value : 10.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.20648224043715843

lambda value : 100.0
Logistic Regression Weighted results:
Accuracy: 65.59%
Error rate: 34.41%
Min DCF for Logistic Regression Weighted: 0.20575364298724952