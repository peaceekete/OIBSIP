# Task 4 — Sentiment Analysis of Nigerian Mobile Banking App Reviews

## Step 1: Introduction & Problem Statement

### Background
Mobile banking has become the primary way customers interact with banks across Nigeria, 
and app store reviews are one of the largest unfiltered sources of customer feedback 
available to a bank. Reading through hundreds of thousands of reviews manually is not 
feasible, so there is a growing need for automated ways to understand what customers are 
saying at scale.

### Problem Statement
Nigerian banks generate large volumes of customer reviews on their mobile apps, but this 
feedback exists as unstructured text with only a star rating attached. Without a way to 
classify the sentiment behind these reviews automatically, banks are limited to reading 
star ratings alone, which don't explain *why* customers feel the way they do, or manually 
reading reviews one at a time, which doesn't scale.

### Project Objectives
To build and compare two supervised machine learning classifiers capable of categorizing 
banking app reviews into Negative, Neutral, or Positive sentiment using the review text 
alone, and to evaluate which model performs more reliably across all three sentiment 
classes, not just the majority class.

### Research Questions
- Can review text alone predict a customer's sentiment category as accurately as their 
  star rating suggests?
- Which text-classification model, Logistic Regression or Multinomial Naive Bayes, 
  performs better when class sizes are imbalanced?
- What role does class imbalance play in a model's ability to detect Neutral sentiment?
- Do star-rating-derived labels always match the sentiment actually expressed in the 
  review text?

## Step 2: Data Sourcing & Understanding

### Data Source
Nigerian Bank App Reviews Dataset, sourced from Hugging Face 
(`Federal-University-Lokoja/Bank-Review`), combining individual review CSVs from 16 
Nigerian banks (Access Bank, GTBank, UBA, Zenith Bank, and others).

### Data Description
- Raw combined dataset: 334,415 rows, 4 columns (`year`, `content`, `score`, `bank`)
- `year`: year the review was posted
- `content`: the raw review text
- `score`: star rating, 1 to 5
- `bank`: which bank's app the review refers to
- Sentiment labels are not provided in the raw data; they are derived from `score` 
  (1-2 = Negative, 3 = Neutral, 4-5 = Positive)

### Ethical Considerations
The dataset consists of publicly posted app store reviews and contains no personal 
identifying information such as names, phone numbers, or account details. No additional 
anonymization was required beyond using the dataset as published.

## Step 3: Methodology & Data Processing

### Data Cleaning
- Removed 123 rows with missing review text (334,415 → 334,292)
- Investigated duplicate reviews and found the vast majority were short (1-2 words: 
  "Good," "Excellent"), which are plausible as independent submissions rather than 
  true duplicates
- Applied a length-gated deduplication: exact duplicates among reviews of 5+ words were 
  reduced to one occurrence; shorter reviews were kept as-is, since collapsing them would 
  have distorted the sentiment distribution
- Final cleaned dataset: 333,821 rows

### Feature Engineering / Text Processing
- Lowercased all text, expanded contractions ("can't" → "cannot"), converted emojis to 
  descriptive text, removed punctuation
- Tokenized cleaned text using NLTK
- Removed stopwords using a customized list that explicitly preserves negation words 
  (no, nor, not, never, cannot), since these carry sentiment-critical meaning
- Applied POS-aware lemmatization, tagging using the full token list (not the 
  stopword-filtered list) so short reviews retain enough grammatical context for accurate 
  tagging

### Exploratory Data Analysis
- Reviewed rating distribution before and after cleaning to confirm deduplication didn't 
  distort the underlying sentiment balance
- Final sentiment class distribution: Negative 30.85% (102,994), Neutral 6.65% (22,212), 
  Positive 62.49% (208,615), visualized as a bar chart
- Generated WordClouds per sentiment class using lemmatized tokens; found that generic, 
  domain-wide terms ("app," "bank," "use") dominate all three classes, while the more 
  informative signal lies in each class's distinctive vocabulary (e.g., "frustrate," 
  "useless" for Negative; "excellent," "reliable" for Positive)

## Step 4: Analysis, Modeling & Results

### Feature Extraction
TF-IDF vectorization, fit on the training set only and used to transform the test set, 
to prevent test-set information from influencing the learned vocabulary. Vocabulary 
capped at 5,000 terms with a minimum document frequency of 10, reducing to 4,492 final 
features after filtering out one-off noise (emoji fragments, rare foreign-language terms).

### Train/Test Split
Split using `GroupShuffleSplit`, grouping by normalized review text so that identical 
review wording could not appear in both the training and test sets, since many short 
reviews repeat verbatim across the dataset. This produced an 80.51% / 19.49% split 
(268,752 training rows, 65,069 test rows), rather than an exact row-level 80/20, because 
whole text-groups were kept together.

### Modeling
Two classifiers were trained on identical TF-IDF features:
- **Logistic Regression** (`class_weight="balanced"`, to counteract class imbalance)
- **Multinomial Naive Bayes** (no built-in imbalance correction)

### Evaluation Metrics

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
|---|---|---|---|---|
| Logistic Regression | 0.78 | 0.64 | 0.67 | 0.6384 |
| Multinomial Naive Bayes | 0.84 | 0.68 | 0.60 | 0.5821 |

Per-class F1 (Negative / Neutral / Positive):
- Logistic Regression: 0.80 / 0.24 / 0.87
- Naive Bayes: 0.82 / 0.03 / 0.90

### Key Findings
- Naive Bayes has the higher raw accuracy, but its Neutral-class recall is only 0.02, 
  meaning it correctly identifies almost none of the actual Neutral reviews
- Logistic Regression's `class_weight="balanced"` setting gives it meaningfully better 
  (though still weak) Neutral-class performance
- Macro-F1, which weights all three classes equally, identifies **Logistic Regression** 
  as the better-performing model in this experiment, since accuracy alone rewards ignoring 
  the minority class
- Confusion matrices confirmed Naive Bayes' Neutral row is almost entirely misclassified 
  into Negative and Positive

## Step 5: Discussion & Business Insights

### Interpretation
Both models can reliably tell clearly positive reviews from clearly negative ones, that 
signal is strong throughout the dataset. Where both models struggle is the Neutral class, 
and this isn't purely a modeling failure. Manual inspection of misclassified Neutral 
reviews found that many contain clearly positive or negative language despite being 
labeled Neutral, because the label came from a 3-star rating, not from the sentiment 
expressed in the text itself. Some reviews also explicitly mention their star rating 
(e.g., "3 stars") within the text, which acts as a target proxy, a shortcut signal tied 
directly to how the label was constructed, distinct from ordinary train/test leakage 
(which was separately guarded against via the grouped split).

### Actionable Recommendations
- Use Logistic Regression over Naive Bayes for this task, given its meaningfully better 
  handling of the minority Neutral class
- Treat Neutral-class predictions with lower confidence than Positive/Negative 
  predictions in any downstream use
- If pursued further, prioritize collecting independently human-annotated sentiment 
  labels rather than relying solely on star ratings

### Limitations
- Sentiment labels are rating-derived, not independently annotated; disagreement between 
  star rating and review text is a primary driver of Neutral-class weakness
- The standard NLTK English stopword list was not customized for Nigerian English or 
  Nigerian Pidgin, which appear in the reviews
- Neutral-class difficulty likely stems from multiple overlapping factors (class 
  imbalance, ambiguous language, the labeling scheme itself) rather than one isolated cause

## Step 6: Conclusion

This project set out to classify Nigerian banking app review sentiment from text alone 
and to compare two models fairly under class imbalance. Logistic Regression was identified 
as the better-performing model based on Macro-F1, the primary metric chosen specifically 
because it doesn't let the majority Positive class mask weaker performance elsewhere. A 
realistic application of this model is triaging incoming reviews, flagging newly posted 
Negative reviews for faster follow-up by a support team, rather than fully automated 
decision-making, given its documented Neutral-class limitations.

No dashboard was built for this task; the analysis and results are contained entirely 
within the accompanying Jupyter notebook.

## Tools
Python, pandas, NLTK, scikit-learn, WordCloud, matplotlib, Hugging Face `datasets`/`huggingface_hub`