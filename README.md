# SD210 Data challenge in Telecom ParisTech

The task was to predict the outcome of patent submissions (accepted or not).

My final approach was a giant Random Forest Classifier (using scikit-learn) of 8000 trees with 36000 leaf each. To train it I used 8 machines of the school, SSH and bash. It took about an hour.

Obviously this approach is not scalable or even usable, but at least it was fun :)

The way to solve this more easily was Gradient Boosted Trees (for example, the famous xgboost-hyperopt combination was used a lot).
