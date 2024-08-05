<h1 align="center"><b>Model Card - Salary Prediction Using Demographic and Employment Data</b></h1>


## Model Details
<li>Random Forest Classifier</li>
<li>Developed by Christopher in 2024</li>
<li>The model utilizes Gini entropy for binary classification of salaries into >=50K or <50K categories based on demographic and employment data.</li>


## Intended Use
<li>Intended to be used for predicting whether an individual's salary is >=50K or <50K based on demographic and employment data.</li>
<li>Can be used by HR professionals, data analysts, sociologists, and economists for analysis and decision-making.</li>
<li>Not intended for use as the sole basis for employment or financial decisions.</li>
<li>Should not be used to make judgments about specific individuals without considering additional context.</li>


## Training Data
<li>The dataset consists of 32,561 rows and 14 columns, with salary as the label.</li>
<li>Data includes demographic and employment information such as age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.</li>
<li>The dataset is split into 80% training and 20% testing using stratified sampling to ensure balanced representation of salary categories.</li>

## Evaluation Data
<li>The evaluation data is a subset of the original dataset, consisting of 20% of the total 32,561 rows, selected using stratified sampling to maintain balanced representation of salary categories.</li>
<li>Data includes the same 14 columns as the training data: age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, native country, and fnlgt.</li>
<li>The evaluation data is used to assess the model's performance on unseen data and to ensure it generalizes well to new inputs.</li>


## Metrics

The model's performance was evaluated using standard classification metrics: precision, recall, and F1 score. On the test set, the model achieved a precision of 0.7099, a recall of 0.6320, and an F1 score of 0.6687. Additionally, the model's performance was assessed on different slices of the test data based on categorical features such as workclass, education, sex, race, and native country. This slice-based evaluation revealed variations in performance across different groups, highlighting areas where the model performs better or worse. For instance, the model showed higher precision and recall for individuals in the 'Federal-gov' workclass and 'Bachelors' education level, while performance was lower for certain racial and nationality groups. This detailed analysis helps identify potential biases and areas for improvement in the model.

## Ethical Considerations

<li><b>Bias and Fairness:</b> Regularly assess and mitigate biases related to sensitive attributes like race, gender, and marital status.</li>
<li><b>Transparency:</b> Provide clear documentation and explainability for the model's decision-making process.</li>
<li><b>Privacy:</b> Protect individuals' data and comply with privacy regulations.</li>
<li><b>Accountability:</b> Regularly audit and update the model to ensure accuracy and fairness.</li>
<li><b>Limitations:</b> Use model predictions as supplementary information, not the sole basis for decisions.</li>
<li><b>Impact on Individuals:</b> Consider broader context to avoid unfair or harmful outcomes.</li>

## Caveats and Recommendations

<li><b>Data Quality:</b> Ensure high-quality, representative training data.</li>
<li><b>Generalization:</b> Be cautious when applying the model to new or diverse populations.</li>
<li><b>Regular Updates:</b> Retrain the model with updated data to maintain accuracy.</li>
<li><b>Holistic Use:</b> Combine model predictions with human judgment and additional context.</li>
<li><b>Monitoring:</b> Continuously monitor performance across demographic groups to identify and address biases.</li>

