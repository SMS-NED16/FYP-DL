# Coursera - Machine Learning - Week 6
# 6.2 - Machine Learning System Design

Will focus on the main issues we may face when designing complex ML systems along with advice for strategizing during ML system design.

## Prioritizing What to Work On
- Suppose we're building a spam classifier. Have a labeled data set of spam/non-spam emails. How to build classifier to distinguish between these two types.
- How to represent the features `X`? 
	- Could come up with a list 100 words that are indicative of spam e.g. buy, deal, discount and non-spam.
	- Encode each email as a feature vector: one-hot encoding that represents whether or not a particular word in the spam corpus appears in the email.
	- `x_j` will be 1 if the email's `j`th word appears in the spam list, and 0 otherwise.
	- Practically, we tend to take the 10k to 50k most frequently occurring words in the training set.
- What is the best use of your time for the classifier to have high accuracy?
	- Collect lots of data: Honeypot projects: create fake email addresses in an attempt to collect spam data. 
	- Develop sophisticated features based on email routing information (email header)
		- Spammers will try to obscure email origins, or send them through fake emails/unusual email routes. 
	- Develop sophisticated features for message body
	- Develop sophisticated algorithms to detect misspellings (spammers do this to intentionally bypass email crawlers and spam classifiers)
- Difficult to say which approach is the best use of your time: teams will randomly fixate on a specific approach.
- **Error analysis** is a systematic approach to finding the best approach for improving ML model performance. 

## Error Analysis
### Andrew Ng's Recommended Approach
- Start with a simple algorithm that you can implement quickly. Implement it and test it on cross-validation data.
- Plot learning curves to decie if more data, more features, etc. are likely to help.
	- Very difficult to tell in advance/in absence of seeing a learning curve what problems exist in the approach and how to improve it.
	- This is a way of avoiding **premature optimization**: let evidence guide design decisions rather than intuition.
- **Error analysis**: Manually examine the examples (in the cross validation set) that your algorithm made errors on. 
	- See if you spot any systematic trend in what type of examples it is making errors on.

### Example - Spam Classifier
- We've built a spam classifier with `m_cv` = 500 i.e. 500 examples in the cross validation set, that gives errors on 100 of these samples.
- Manually examine the 100 errors and categorize them based on
	- What type of email it is
	- What cues you think would have helped the algorithm classify them correctly
- Findings
	- 12 misclassified emails are from pharmaceuticals, 4 are replicas or fakes, 53 are password-stealing/phishing emails, and 31 are other.
	- Suggests that we should focus more on phishing emails because the algo performs most poorly on it.
- Look at feature activations: what features were most useful in helping the algo classify emails?
- Need to figure out what are the most difficult examples for the algo to classify - quick implementation helps do this and tailor the final product around them,

### Example - Stemming and Numerical Evaluation
- Very helpful to have a way of evaluating the performance of the learning algo in terms of a single real number. 
- E.g. should discount, discounts, discounted, and/or discounting be considered the same in classifying an email as spam or non-spam?
- In NLP, we use stemming software e.g. Porter stemmer to treat all words with same root word as identical for classification purposes.
- Stemming software can often hurt classification accuracy, but this may not be apparent from error analysis alone.
- Instead if we had a numerical evaluation metric e.g. the cross-validation classification error, we could test performance with/without stemming: if stemming decreases error -> stemming might be worthwhile.
- For this specific problem, there was a very straightforward real number evaluation metric. This is not always the case! 
- But generally, numerical evaluation can immediately tell whether or not a design decision improved an ML system's performance.

## Error Metrics for Skewed Classes
- Skewed classes: specific class (or classes) are either underrepresented or overrepresented in the data set. 

### Example - Cancer Classification
- Trained a logistic regression classifier which predicts `y` = 1 (malignant) or `y` = 0 otherwise.
- We get 99% classification accuracy on the test set, but only 0.50% of the training set samples actually had cancer.
- The dummy algorithm
`
function y = predictCancer(x)
	y = 0; % ignore x entirely! Just predict 0 i.e. benign cancer, regardless of x
`
actually gets a 0.5% error, and so technically does better than the actual classifier!
- When the ratio of positive to negative examples is extreme (y = 1 << y = 0) -> case of skewed classes. 
- With skewed classes, numerical evaluation is no longer as straightforward as using classification accuracy.

### Precision/Recall
- Improved numerical evaluation metrics for skewed classification problems. 
- Derived from a confusion matrix - a 2 x 2 matrix or table which tallies
	- True positives: Predicted and actual classes are 1.
	- False positives: Predicted class 1, actual class 0. 
	- True negatives: Predicted and actual classes are 0.
	- False negatives: Predicted class 0, actual class 1. 
- Precision: Of all patients where we predicted `y = 1`, what fraction actually has cancer?
	- Precision = True Positives / Number we predicted as positives
	- **Precision = TP / (TP + FP)**
- Recall: Of all patients that actually have cancer, what fraction did we correctly detect as having cancer? 
	- Recall = True Positives / Number of actual positives
	- **Recall = TP / (TP + FN)**
	- False negatives: people we misclassified as not having cancer when they actually did have cancer. 
- If classifier predicts no one has cancer, then **recall** will be 0. 
- Even for skewed classes, it is difficult for an algo to get a high precision and recall.
- Conventionally, `y = 1` is assigned to the rare class that we are trying to detect. 

## Precision-Recall Tradeoff
- Logistic regression classifier for cancer: `h_theta(x)` predicts `y = 1` is >= 0.5, or `y = 0` < 0.5.
- We want to be more stringent: we want to be more certain that the person has cancer, because otherwise they'd have to go through an expensive and painful treatment.
- One way of doing this is to raise the threshold for `h_theta(x)` - it predicts `y = 1` if output is >= 0.7.
	- Have higher precision: a higher proportion of predicted cancer patients will be TPs.
	- But lower recall.
- If we want to avoid false negatives (patient actually has cancer, but we fail to tell them)
	- In many ways, false negatives are more dangerous in this case: patients won't get treatment in time.
	- Want to avoid missing too many cases of cancer. 
	- Rather than setting a high probability threshold, would set a lower threshold: e.g. 0.3
	- Will be more conservative: even if there's a 30% probability of cancer, we will recommend treatement.
	- Higher recall: will be reducing false negatives.
	- Low precision: False positives will increase.
- **There is a tradeoff here: can have high precision or high recall, but not both**, depends on the threshold. 
- Can plot the precision-recall curve at different threshold values. 

### F1 Score - Comparing P/R Ratio
- How to compare precision/recall numbers? Which combination is best?
- F1 Score is a single numerical evaluation metric for assessing the quality of a P/R ratio.
- One approach: average precision and recall = (P + R)/ 2
	- Not a good approach. Precision/recall values can have a large disparity, and the average is susceptible to outliers. 
	- E.g. if P = 0.02 and R = 1.0, the avg is 0.51.
		- This may seem like a higher avg, but what good is a classifier that has ~0 precision (%age of TPs)?
- **F1 Score** = 2 * P * R / (P + R)
- Gives the lower value of precision/recall a higher weight, which prevents the higher of the two values from skewing the evaluation metric.
- Many different possible formulae for combining precision and recall, but historically F1 score is most commonly used. 
- If P = 0 or R = 0, F_score = 0 (worst theoretical P/R tradeoff)
- If P = 1 and R = 1, F_score = 1 (perfect theoretical precision and recall).
- If the goal is to set a good threshold for deciding between precision and recall, try a different range of thresholds and find the one with the best F1 score. 

## Data for Machine Learning
- How much data to train on?
- Under specific conditions, getting a lot of data and training on a certain type of learning algo can be an effective way to get good performance. 
- Banko and Brill (2001) conducted a supervised learning study on classification of easily confusable words (two/too) using different algorithms.
	- Most algos give remarkably similar performance.
	- As the training set size (in millions) increases, the performance of the algorithms monotonically increases.
	- Even an "inferior" algorithm can beat a "superior" algorithm if provided enough training examples.
- **It's not who has the best algorithm who wins; it's who has the most data**.

### Large Data Rationale?
- When does data >> algos?
- Assume that features `x` have sufficient information to accurately predict `y`. 
	- Can a human expert confidently predict the value of `y` with the same features that are given to the algorithm. 
	- E.g. features `x` are able to capture the words in the sentence that could help infer context for too/two. 
	- Counterexample: predicting the price of a house **only** using its size in square feet: not enough information to make accurate prediction.
- Assume also that we also have a very large number of parameters.
	- Low bias algorithms.
	- `J_train` will be small.
- Assume the dataset is very large and so the algorithm is unlikely to overfit.
	- `J_train` is close to `J_test`.
	- This implies the test set error will also be small. 
- We are simultaneously solving the high bias and high variance problems - such an algo will have low test error. 