# Evaluation Report — AI-Powered Sentiment Analysis

## Objective
Classify customer reviews as Positive, Negative, or Neutral using review text.

## Dataset audit
- Rows: 10,000
- Columns: 5
- Missing values: 0
- Exact duplicate rows: 0
- Unique review text strings: 300
- Sentiment distribution: {'Positive': 3334, 'Negative': 3333, 'Neutral': 3333}

**Important data-quality finding:** 10,000 rows contain only 300 unique review text strings. A standard random row split would place identical strings in train and test, inflating results. We used a group-aware split by `review_text`, and grouped cross-validation, to prevent exact text overlap across evaluation folds.

## Experimental design
- Group-aware 80/20 split by exact review text (random seed 42)
- TF-IDF features
- Baselines: Logistic Regression, Multinomial Naive Bayes, Linear SVM
- Grid Search: Logistic Regression; 3-fold StratifiedGroupKFold
- Selection metric: macro F1-score

## Baseline comparison
                  Model  Accuracy  Precision (macro)  Recall (macro)  F1-score (macro)
    Logistic Regression       1.0                1.0             1.0               1.0
Multinomial Naive Bayes       1.0                1.0             1.0               1.0
             Linear SVM       1.0                1.0             1.0               1.0

## Tuned model
- Best parameters: {'classifier__C': 0.5, 'tfidf__ngram_range': (1, 2)}
- Best grouped cross-validation macro F1: 1.0000

## Held-out test metrics
 Accuracy  Precision (macro)  Recall (macro)  F1-score (macro)
      1.0                1.0             1.0               1.0

## Classification report
```text
              precision    recall  f1-score   support

    Negative       1.00      1.00      1.00       727
     Neutral       1.00      1.00      1.00       761
    Positive       1.00      1.00      1.00       520

    accuracy                           1.00      2008
   macro avg       1.00      1.00      1.00      2008
weighted avg       1.00      1.00      1.00      2008

```

## Confusion matrix
Rows are true labels; columns are predictions. Order: Negative, Neutral, Positive.
```text
[[727   0   0]
 [  0 761   0]
 [  0   0 520]]
```

## Interpretation and limitations
The measured score is high, but it must be interpreted cautiously because the dataset has only 300 distinct review texts. Even with grouped evaluation, those templates may have highly class-specific wording and may not represent naturally occurring customer reviews. The model should be tested on a genuinely independent set of new reviews before making real-world performance claims. The deployed pipeline is refit on all provided rows using the selected hyperparameters; the held-out metrics above come from the group-aware evaluation before that final refit.
