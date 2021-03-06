# FYP-DL: Applications of Neural Networks for Anomalous Energy Consumption Detection

Code for tutorials, tests, and experimentation done as part of a Deep Learning-based undergraduate Final Year Project at the department of Electrical Engineering, NEDUET.

## Project Summary
Using supervised deep learning to systematically create, test, and optimise neural networks for detecting anomalous energy consumption patterns in smart meter data. Training using a [publicly available, labeled dataset published by the State Grid Corporation of China](https://github.com/henryRDlab/ElectricityTheftDetection) which spans 13 months of daily kWh smart meter measurements of ~42k residential consumers.

More details are in the [Project Proposal](./FYP-Documents/Proposals/fyp-proposal-gh.pdf).

## Progress
|Evaluation|Date|Presentation|Report|Summary|
|:--------:|:--------:|:--------:|:--------:|--------------------|
|Final|October 2020|[Final Presentation](./FYP-Documents/eed-group13-final-presentation.pdf)|[Final Report](./FYP-Documents/eed-group13-final-report.pdf)|CNN, WDNN, Class Imbalance, Tuning, Deployment|
|Midyear|March 2020|[Midyear Presentation](./FYP-Documents/eed-group13-midyear-presentation.pdf)|[Midyear Report](./FYP-Documents/eed-group13-midyear-report.pdf)|LogReg, SVM, Random Forest, WNN, Tuning|

## Results
### Improvement of up to 6% on benchmark's test set ROC AUC 
<img src="./FYP-Documents/Images/fyp-final-results.png" alt="FYP Results - ROC AUC" align="center">
<img src="./FYP-Documents/Images/fyp-final-results-pct.png" alt="FYP Results - ROC AUC Improvement" align="center">

### Successfully deployed and used for inference as API on AWS Sagemaker.
<img src="./FYP-Documents/Images/fyp-aws-api.png">
<img src="./FYP-Documents/Images/fyp-aws-req-res.png">

## Group Members
All group members are final year undergraduates from Section D, Batch 2016-17 at the Department of Electrical Engineering, NEDUET.

|No.|Roll Number|Name|
|:--------:|:-----------:|------------|
|1|EE-16177|Muhammad Waleed Hasan (Leader)|
|2|EE-16163|Saad Mashkoor Siddiqui|
|3|EE-16164|Faiq Siddiqui|
|4|EE-16194|Syed Abdul Haseeb Qadri|

## Advisors

|    |Name|Designation|Organization|
|:-----:|:------:|:-----:|:-------:|
|**Internal**|Dr. Mirza Muhammad Ali Baig|Assistant Professor|Department of Electrical Engineering, NEDUET|
|**External**|Mr. Shahzeb Anwar|Infrastructure Analyst|ENI Pakistan|
