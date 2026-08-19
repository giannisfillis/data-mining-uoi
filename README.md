# Data Mining - Academic Year 2024-2025

[![Course](https://img.shields.io/badge/Course-Data_Mining_(MYE012)-blue.svg)]()
[![Institution](https://img.shields.io/badge/Institution-University_of_Ioannina-red.svg)]()
[![Language](https://img.shields.io/badge/Language-Python-orange.svg)]()
[![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy%20%7C%20Scikit--Learn-lightgrey.svg)]()

This repository contains the laboratory assignments for the Data Mining course at the University of Ioannina, Department of Computer Science & Engineering. 

The projects focus on exploratory data analysis (EDA), statistical hypothesis testing, recommender systems, clustering, natural language processing (NLP), and graph-based analysis. All implementations were done in **Python**, utilizing vectorized operations and sparse data structures for performance.

**Team Members:**
* Giannis Fillis, AM: 5380
* Konstantinos Zois, AM: 5226

---

## 📖 Table of Contents
1. [Project 1: Exploratory Data Analysis & Statistical Testing](#project-1-exploratory-data-analysis--statistical-testing)
2. [Project 2: Recommender Systems (SVD & Collaborative Filtering)](#project-2-recommender-systems-svd--collaborative-filtering)
3. [Project 3: Clustering, Text Classification & Graph Recommendations](#project-3-clustering-text-classification--graph-recommendations)

---

## 📊 Project 1: Exploratory Data Analysis & Statistical Testing

### Overview
Comprehensive data exploration, cleaning, and hypothesis testing on a large collection of movies to discover underlying correlations and historical trends.

### Dataset
* **Kaggle Movies Metadata** (`movies_metadata.csv`, `credits.csv`).

### Implementation Details
* **Data Cleaning & Wrangling:** Handled missing values, incorrect formats, and string-parsing anomalies using Pandas.
* **Distribution Analysis:** Analyzed the distribution of `vote_counts` utilizing linear and exponentially sized bins, cumulative frequency, and Zipf's law plots.
* **Correlation Analysis:** Investigated the relationship between movie runtime, budget, revenue, and average rating using scatter plots and Pearson correlation coefficients (with p-values).
* **Temporal Analysis:** Visualized the evolution of movie quality across decades, calculating 95% confidence intervals using Seaborn.
* **Hypothesis Testing:** 
  * Extracted and matched directors' genders with movie genres using contingency tables, evaluating statistical significance via $\chi^{2}$-tests and Lift metrics.
  * Compared the financial and rating profiles of "Action" vs. "Drama" movies using t-tests and bar plots.
* **Bonus Implementations:**
  * Developed a bias-correction method to address the sparse representation of older movies in temporal quality charts.
  * Formulated and evaluated custom non-trivial statistical hypotheses.

---

## 🤖 Project 2: Recommender Systems (SVD & Collaborative Filtering)

### Overview
Implementation of various personalized movie recommendation algorithms from scratch, prioritizing performance via sparse matrices and purely vectorized computations without iterative looping.

### Dataset
* **Kaggle Small Ratings Dataset** (`ratings_small.csv`), featuring 100,000 ratings from 673 users across 9,066 movies. Data was split into 90% training and 10% testing.

### Implementation Details
* **Baselines:** Implemented User Average (UA) and Item Average (IA) prediction models.
* **Singular Value Decomposition (SVD):** Evaluated rank-$k$ approximations (for $k \in [1, 20]$) using sparse matrices to predict unrated movies, analyzing the Root Mean Square Error (RMSE) convergence.
* **User-Based Collaborative Filtering (UCF):** Built a highly optimized UCF algorithm utilizing cosine similarity. Extracted $k$-nearest neighbors to predict ratings.
* **Bonus Implementations:**
  * **Mean-Centered UCF:** Improved the baseline UCF by centering ratings to predict deviations from the user's mean, compensating for optimistic/pessimistic raters.
  * **Item-Based Collaborative Filtering (ICF):** Transposed the approach to recommend movies based on item similarity rather than user similarity.

---

## 🕸️ Project 3: Clustering, Text Classification & Graph Recommendations

### Overview
Applied unsupervised machine learning to group movies, supervised NLP to classify movie genres based on textual summaries, and graph algorithms to enhance recommendation accuracy.

### Dataset
* Integrates `ratings_small.csv`, `links_small.csv`, and `movies_metadata.csv`.

### Implementation Details
* **Movie Clustering:** 
  * Clustered movies based on user rating behaviors using K-Means and Agglomerative Clustering.
  * Evaluated the optimal number of clusters ($k \in [2, 20]$) via Sum of Squared Errors (SSE) and Silhouette coefficients.
  * Validated cluster logic by analyzing genre distributions and Lift metrics. Re-ran experiments reducing dimensionality with PCA (50 latent factors).
* **Text Classification (NLP):**
  * **Goal:** Predict 3 distinct genres (War, Music, Animation) from movie overviews.
  * **Feature Extraction:** Evaluated **TF-IDF** vectorization versus Pre-trained **Word2Vec** (Gensim) embeddings.
  * **Models:** Trained and compared Decision Trees, K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), and Multilayer Perceptrons (MLP).
  * **Evaluation:** Used strict 5-fold cross-validation, reporting Accuracy, Precision, Recall, and Confusion Matrices.
* **Graph-Based Recommender (Personalized PageRank):**
  * Replaced the standard cosine similarity in UCF with a graph-based approach.
  * Constructed a User-User similarity graph keeping only the top-$n$ nearest neighbors per user as weighted edges.
  * Executed **Personalized PageRank (PPR)** algorithms to determine indirect user similarities and recalculated rating predictions, comparing RMSE improvements against Project 2.
* **Bonus Implementations:**
  * Augmented the text classifiers with additional structural/numerical features from the movie metadata to boost performance.
  * Experimented with various restart probabilities in the PageRank algorithm to fine-tune similarity propagation.
