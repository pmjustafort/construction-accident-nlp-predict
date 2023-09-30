# Predicting Construction Accident Severity with Natural Language Processing and Data-Driven Recommendations.

![402316fc1df0d211682a585969559bfa](https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/16bce4a6-fc71-4d04-bea1-eae1d3af7083)

## Overview
This project seeks to create a predictive model for construction accident severity, with the ultimate goal of providing recommendations to enhance on-site safety. Initially, we developed a baseline model, utilizing only numerical and categorical features. As we delve deeper, we exploit the power of Natural Language Processing (NLP) in the text features. This rich textual data is then transformed into a numerical format through TF-IDF techniques. When integrated with the original dataset's attributes, this combined data set equips us to better predict the  'Degree of Injury' and thereby make informed safety recommendations.


## Motivation

![number-and-rate-of-fatal](https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/813dfd39-f488-4214-a96c-b56779d5337f)

The construction sector's safety concerns have been thrown into sharp relief by the recent statistics from the U.S. Bureau of Labor Statistics. In 2021 alone, the private construction sector reported a harrowing 986 fatalities. This figure not only surpasses the combined total of the transportation and warehousing sectors, which reported 976 fatalities, but is also more than double the fatalities in the agriculture, forestry, fishing, and hunting sectors combined, which reported 453 fatalities.

The ramifications of construction accidents extend far beyond these grim numbers. They lead to significant economic impacts due to halted projects, legal claims, and compensation payouts. The environmental consequences can be severe, with accidents sometimes leading to spills, contamination, or damage to the local ecosystem. Delays in construction timelines not only escalate project costs but also inconvenience communities awaiting the completion of infrastructure or housing projects.

However, the most profound impact is undoubtedly on the human front. The emotional and social toll on families who lose a loved one or see them suffer severe injuries is immeasurable. Communities grieve, and the fabric of society is weakened every time safety is compromised.

The pressing need to enhance safety protocols and measures in the construction industry becomes evident. With construction recognized as one of the most hazardous industries, there's a dire need to utilize data-driven insights to mitigate these risks and protect the lives of construction workers. As the industry evolves, the focus must firmly be on ensuring that every worker returns home safely at the end of the day.

## Data Sources
Our primary dataset for this project is sourced from OSHA's (Occupational Safety and Health Administration) records of construction fatalities, which we obtained from [Kaggle](https://www.kaggle.com/datasets/ruqaiyaship/osha-accident-and-injury-data-1517). This dataset comprises 4847 records spanning from July 2015 to August 2017. Each record is described by 29 distinct features, offering a detailed account of various construction accidents, from textual event descriptions to factors leading up to the incident.
This rich dataset provides a comprehensive look into accidents within the construction sector over a two-year period, setting the foundation for our predictive modeling and subsequent safety recommendations.

## Feature Engineering
Handling Missing or Placeholder Data:
•	Placeholder Identification: During our preliminary assessment, we identified multiple placeholder entries, notably the "0" value, in significant columns such as 'Construction End Use', 'Building Stories', and 'Project Cost'.
•	Handling Strategies: We evaluated and applied transformation and imputation strategies to handle these placeholder entries, ensuring that any potential model bias stemming from these could be minimized.

Textual Data Processing:
•	Combination of Descriptive Text: Recognizing the value held within textual columns, we combined 'Abstract Text', 'Event Description', and 'Event Keywords' into a single 'combined_text' column. This step aimed to capture and unify all the descriptive information related to each event.
•	Text-to-Numerical Transformation: We employed the TF-IDF Vectorizer to convert this textual data into a numerical format. This transformation quantified the importance of words in the context of the entire dataset, making the textual data digestible for our predictive models.

Categorical Encoding:
•	Feature Selection for Encoding: Some columns, like 'nature_of_inj' and 'Nature of Injury', offered overlapping information. We prioritized more descriptive versions of such features for our analysis.
•	Encoding Process: The selected categorical features were then encoded into numerical values, making them suitable for modeling. This ensured that all categorical variables were interpreted correctly by our machine learning algorithms.


## Results
We initially embarked on our modeling journey with a baseline Logistic Regression model, leveraging only the numerical and categorical features. This approach provided us with an understanding of our data's intrinsic predictability and set a benchmark for further refinements.
The baseline model showcased a compelling initial performance, with a mean CV accuracy of 91.20% (±1.59%), signifying the substantial predictive power of our initial feature set. Precision and recall values, hovering around 92.75% (±1.48%) and 92.89% (±1.81%) respectively, highlighted the model's balanced predictions. This equilibrium was further emphasized by an F1-Score of 92.80% (±1.32%), and the model's adeptness in class differentiation was confirmed by a standout ROC-AUC score of 96.87% (±0.53%).
Upon integrating the latent insights from the 'Abstract Text', 'Event Description', and 'Event Keywords' columns, our Logistic Regression model saw a substantial enhancement in its predictive capability. The mean CV accuracy jumped to 95.54% (±0.77%). Precision rose slightly to 95.36% (±0.62%), while recall saw a notable increase to 97.38% (±1.02%), indicating more accurate positive case identifications without significantly raising false positives. This refined model's prowess was further validated by an F1-Score of 96.35% (±0.63%) and an exceptional ROC-AUC of 98.25% (±0.24%).

![Numeric:Categorical Model And Text Incorporated Model Comparison](https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/cbbdee04-5ab4-47fc-8fb3-8c7023cc51e8)

The comparison between the two models painted a vivid picture. While our baseline already exhibited commendable performance, the integration of textual data transformed our model into a far more potent tool. The significant leaps in accuracy, recall, and the ROC-AUC score stood testament to the richness and importance of textual information in our dataset. This venture into Natural Language Processing not only bolstered our model's predictive capability but also paved the way for more in-depth insights and nuanced interpretations in our further analyses
To refine our model further, we evaluated the impact of severity-indicative words like killed, died, and dead, by introducing custom stopwords. When juxtaposed, the differences between the models – one with all words and the other sans these indicative words – proved minuscule. While the all-inclusive model demonstrated marginally superior results across metrics, the gap was negligible. Given our commitment to clarity in predictions and comprehension of severity determinants, we've chosen to retain the model incorporating all words. This strategy not only maximizes data utilization but also fosters a transparent modeling procedure.

<img width="1048" alt="Model With Severity Words And Model Without Severity Words Comparison" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/0e84665f-1bae-445e-81fc-255e5cd45ad3">

As we look towards future enhancements, our attention gravitates towards algorithms like Random Forest and XGBoost, and the concept of a stacked model. With its reputation for robustness, we hypothesize that the Random Forest algorithm might excel by optimally leveraging the heterogeneity in our data.

<img width="1015" alt="Accuracy Comp" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/abb2e253-9f31-415a-b3f1-acc5634c740a">

<img width="1015" alt="Precision Com" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/c56c881e-d7d7-485d-bda2-771d4cd66171">

<img width="1015" alt="Recall COm" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/9a1949f7-ae24-43d1-ad94-96bf0feedb2c">

<img width="1015" alt="ROC Comp" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/81aec213-609a-4060-9470-3abf3c2eb527">

Both Stacekd models, tuned an non-tuned emerge as our frontrunners. These models distinctly overshadow their counterparts - the standalone Logistic Regression, Random Forest, and XGBoost - across all evaluation metrics. This solidifies the potency of ensemble stacking, a technique adept at amalgamating strengths from various models. While the performance gap between the stacked models and XGBoost remains thin, the scales tilt slightly in favor of the stacked ensemble. Furthermore, a head-to-head between the tuned and non-tuned stacked models reveals marginal performance disparities. However, with efficiency in the modeling process at the forefront, and a smidgeon of superiority in precision, our choice solidified around the non-tuned Stacked Model for ensuing endeavors. This choice not only upholds model efficacy but also envisions a smooth, effective modeling journey.

While our stacked model exhibited stellar cross-validation results, a significant divergence was observed when subjected to testing data with very lower scores such as an Accuracy of 80.87%, a Precision of 76.23%, a striking Recall of 99.83%, an F1-Score of 86.45%, and an ROC-AUC of 98.45%. This shift hints at the potentiality that our base models within the stacked configuration may have had strongly correlated predictions, diminishing the inherent advantages of the stacking mechanism. Such correlation could stifle diversification in the collective predictions, an attribute ideally sought in stacked models. Recognizing this, we surmise that enhancing the individual models might bear more fruitful outcomes. Consequently, our sights are now set on refining the XGBoost model, our second ace in the pack. By meticulous parameter tuning, our aspiration is to elevate its efficacy and thereafter gauge its resilience against the testing data.

Our meticulously tuned XGBoost model emerges as a formidable contender, manifesting stellar performance across the board. With a Mean CV Accuracy of 98.11%, it is not only highly accurate but also consistently reliable. Further, the precision, standing at 98.11%, and an astoundingly high recall of 98.81% exemplify its balanced prowess in both predicting and capturing true positives. An F1-Score of 98.46% further corroborates this equilibrium, showcasing the Model's Harmonized strength in precision and recall. Notably, the model's ability to discern between the classes is underlined by an impressive ROC-AUC score of 98.95%. This paramount performance of the XGBoost, post hyperparameter tuning, draws it very close to our earlier stacked model, making it an equally, if not more, viable tool for predicting construction accident severity in our dataset.

<img width="1015" alt="Stack ANd XGB CV" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/eee013db-1b15-46eb-b27d-e82604ac95b7">

Upon comparing the metrics of the Non-Tuned Stacked Model with the Tuned XGBoost in cross-validation, both models exhibit commendable performances. While the Stacked Model slightly edges out in most metrics, it's crucial to remember that this model didn't generalize well to the testing data. On the other hand, the Tuned XGBoost demonstrates slightly lower, yet competitive, cross-validation scores. Given the suboptimal performance of the Stacked Model on unseen data, the Tuned XGBoost model becomes our primary choice. Its reliable performance in cross-validation, combined with its expected better generalization to new data, make it a more trustworthy model for future predictions.

Having established our preference based on cross-validation performance, it's imperative to validate our model's efficacy on an external set. Let's delve into how these models fare when challenged with the testing data.

<img width="1015" alt="Stack And XGB Test" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/6206ef72-434e-4ca5-84b1-b00b00c824d5">

The performance metrics of our optimal model underscore its exemplary capability in predicting outcomes. Achieving an accuracy of 98.35%, it assures that the vast majority of its predictions are accurate. Precision, standing at 98.32%, signifies the model's adeptness at minimizing false positives, ensuring that the positive predictions made are, in fact, correct. A commendably high recall of 98.98% illustrates the model's strength in identifying almost all actual positive cases, minimizing false negatives. The harmonious balance between precision and recall is further echoed by an F1-Score of 98.65%. Lastly, an ROC-AUC of 98.77% reiterates the model's robustness in effectively distinguishing between the classes. As we delve deeper into our model's evaluation, the upcoming confusion matrix, performance curves, and feature importance will provide even more granular insights into its capabilities and behavior.

<img width="586" alt="CM" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/eed7c647-ab84-4a28-918b-4d51b53b4b80">

The analysis of our model's predictions unveils a high degree of accuracy. Out of the predictions, 585 events were rightly identified as Fatal, illustrating the model's efficiency in detecting the most severe cases — the true positives. Furthermore, it accurately discerned 366 events as Non-Fatal, denoting the true negatives. On the flip side, the model was not without its slight discrepancies. It erroneously categorized 10 events as Fatal when they were, in reality, Non-Fatal, manifesting as false positives. Similarly, there were 6 instances, or false negatives, where events were predicted as Non-Fatal despite being genuinely Fatal.

Next, we present the Receiver Operating Characteristic (ROC) curve, a graphical representation that illustrates the diagnostic ability of our model across various threshold settings. 

<img width="982" alt="ROC XGB" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/e6e7c406-daf8-4123-b578-0b3b537562d3">

Between sensitivity (true positive rate) and specificity (false positive rate). An area of 0.98, nearing the perfect score of 1, signifies that our model possesses a high discriminative power, and is capable of distinguishing Fatal from Non-Fatal events with exceptional accuracy. Such a high area under the curve is indicative of a model that not only predicts outcomes accurately but also minimizes the chances of misclassification.
Let’s see the Precision-Recall (PR) curve. This graph emphasizes our model's skill in forecasting the positive (Fatal) class by juxtaposing precision against different levels of recall. Given its sensitivity to class imbalances, the PR curve provides an insightful lens into the model's precision and its ability to distinguish events effectively.

<img width="999" alt="Precision Recall Curve" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/4e0719f0-ee94-4cd0-b45a-0c810ff1c9cb">

The precision-recall curve commences with perfect precision at zero recall, then maintains a steady and high precision as recall increases. Remarkably, the model maintains a precision above 0.97 even as recall approaches 0.98. Such behavior indicates the model's robustness in preserving high precision even while capturing most of the positive cases. The AUC-PR value of 0.99 further reinforces the model's exemplary performance across different decision thresholds.

Now, diving into the intrinsic characteristics that drive our model's decisions, we reveal the top 10 features pivotal to its predictions.

<img width="1010" alt="Top 10 Features" src="https://github.com/pmjustafort/construction-accident-nlp-predict/assets/137816262/6e591016-6fbf-48c6-95d1-8813b8393ab5">

These features, ranked by their importance, shed light on the key factors that the model perceives as influential when predicting the severity of construction accidents. From specific words or phrases in the textual data to distinct event types and natures of injuries, these indicators offer a holistic view of the conditions and situations that correlate most strongly with fatal outcomes.

## Conclusion 
Heavy machinery operations, particularly those involved in excavation and digging, have led to severe injuries and fatalities, highlighting the critical need for safe machine operation. Time-related factors, including prolonged exposure and delays in medical treatment, play a substantial role in determining the severity and outcome of incidents. It is evident that preventive measures and prompt responses are crucial in minimizing risks.

Incidents involving severe falls, strikes, amputations, and crushing injuries constitute a significant portion of the dataset, underscoring the paramount importance of preventive measures and comprehensive training programs. 

Ductwork, often associated with HVAC systems, has been linked to various incidents, ranging from electrical hazards to falls. This emphasizes the necessity for careful handling and rigorous safety checks in all aspects of construction.

Tasks performed at elevated heights, whether through ladders or machinery, have resulted in fatal injuries, spotlighting the need for stringent safety protocols in these scenarios. Electrocution and electrical shocks emerge as some of the deadliest incidents, underscoring the critical importance of electrical safety and the need for robust precautions and training programs.

These quantitative and contextual insights, amalgamated with the predictive power of our Tuned XGBoost predictive model exhibiting exceptional predictive capabilities, serve as a powerful framework for enhancing decision-making in workplace safety. Our model achieves an accuracy of approximately 98.35%, distinguishing itself with an impressive precision of 98.32%, a recall of 98.98%, and an F1-score nearing 98.65%. The ROC-AUC, at 98.77%, further showcases the model's effectiveness in classifying 366 non-fatal accidents, 585 fatal accidents, with only marginal misclassifications. By leveraging the lessons gleaned from these key features, organizations can better prioritize safety measures, fostering a culture of proactive risk mitigation and ultimately safeguarding the well-being of their workforce.


## Recommendations
- Safety Training: Ensure all workers undergo comprehensive safety training. This should be routinely updated and tailored to the specific equipment and tasks they'll be handling.

- Electrical Safety: Given the high number of electrocutions, electrical safety should be emphasized. This includes proper grounding, insulation, and the use of protective gear.

- Heavy Machinery Operations: Emphasize rigorous training for operators of heavy machinery, like excavators and forklifts. Additionally, establish a clear protocol for others on the site to ensure they are at a safe distance.

- Work Hour Regulation: Monitor and limit the number of continuous hours an employee can work, especially in physically demanding roles or extreme conditions, to prevent fatigue-related accidents.

- Immediate Medical Attention: Implement a protocol for timely medical intervention in the event of injuries, no matter how minor they seem.

- Fall Protection: Ensure all elevated tasks have appropriate fall protection measures, from harnesses to guardrails.

- Equipment Maintenance: Regularly maintain and inspect equipment, especially those associated with lifting and elevating tasks, to ensure they are in proper working condition.

- Personal Protective Equipment (PPE): Ensure all workers have access to and use the right PPE for their tasks, from helmets to protective gloves.

- Safety Audits: Conduct regular safety audits of worksites to identify and mitigate potential hazards proactively.

- Specialized Training for Specific Areas: Areas like kitchens, although not traditional construction zones, should be included in safety training, given the unique hazards they present.

In addition to enforcing OSHA and state regulations, by implementing these recommendations, companies can ensure the safety and well-being of their employees, reduce the risk of accidents, and foster a culture of safety first. These proactive measures serve as a vital layer of protection and go hand-in-hand with existing safety standards, providing an extra level of assurance in construction site safety.


## Next Steps
- Feature Emphasis: Focus on key features such as '000', 'kitchen', 'digging', and specific injury types ('Serious Fall/Strike', 'Amputation, Crushing') for in-depth safety audits and investigations.

- Further Exploration: Feature like 'kitchen', is intriguing and warrant a deeper dive into the data to ascertain is significance. A better understanding of the context behind it might reveal novel insights about construction accidents.

- Model Deployment: Integrate our high-performing Tuned XGBoost model into real-time monitoring systems to proactively identify high-risk incidents and enable timely interventions.

- Continuous Update: Regularly update and retrain our model with new data to ensure its predictions remain accurate and relevant.

- Data Augmentation: Expand our dataset with more accident records and aim to acquire more recent and extensive data sources. Ensure data quality, preprocess new data, and refine feature engineering techniques.


## Repository Structure
