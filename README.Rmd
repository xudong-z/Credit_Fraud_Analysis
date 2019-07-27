# Credit Card Fraud Detection (Classification Modeling)

### Dashboard link: 
https://xudong-z.shinyapps.io/credit-card-fraud-analysis/

#### 1. EDA on each variables
<img width="1227" alt="1-EDA" src="https://user-images.githubusercontent.com/20660492/61994425-a758f800-b0ac-11e9-8c1d-20329717cb29.png">


### 2. Classification models evaluation (based on Area under ROC and Precision-Recall curve)
candidate models: Logit, Random Forest, KNN, Neural Network, 
<img width="1167" alt="2-evaluation" src="https://user-images.githubusercontent.com/20660492/61994426-a758f800-b0ac-11e9-9fcb-fef98d0ba12d.png">

### 3. Detection Strategy Analysis 
 - for transactions with small amount, should take radical detection, the Recall score in PR curve matters more
 - for transactions with large amount, should take conservation detection, the Precision score in PR curve matters more
<img width="913" alt="3-stategy analysis" src="https://user-images.githubusercontent.com/20660492/61994427-a758f800-b0ac-11e9-8ebe-44eac708ea25.png">

### 4. Best model - Random Forest in use
 - allow users to customize the select RF model by setting the oversampling level, detection accuration threshold
<img width="1159" alt="4-Random Forest application" src="https://user-images.githubusercontent.com/20660492/61994428-a7f18e80-b0ac-11e9-9f9b-4b0663b0f587.png">

### 5. Actional Suggestions
<img width="900" alt="5-suggestion" src="https://user-images.githubusercontent.com/20660492/61994429-a7f18e80-b0ac-11e9-9d81-b0cd80ef7b5a.png">
