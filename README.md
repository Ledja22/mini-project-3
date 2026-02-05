# Marketing Campaign Customer Segmentation Analysis

## Problem Description

**Business Problem:** Identify and segment customers based on their purchasing behavior and demographics to enable targeted marketing strategies and personalized customer engagement.

**Why Unsupervised Learning?** Unlike supervised learning where we know the target labels, we have no predefined customer segments. Clustering discovers natural groupings in the data, revealing hidden patterns that can drive business decisions without labeled training data. This approach is ideal for customer segmentation where we want to discover meaningful patterns rather than predict a known outcome.

---

## Dataset Description

**Source & Size:** Marketing campaign dataset containing 2,240 customer records with 29 features  
**After Preprocessing:** 1,999 customers (outliers removed, features selected)

**Key Features:**
- **Demographics:** Year_Birth, Education, Marital_Status, Income, Kids at home, Teens at home
- **Spending:** MntWines, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds
- **Behavior:** NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, NumDealsPurchases
- **Response:** Campaign acceptance indicators, Response flag

**Preprocessing Steps:**
1. Removed missing values
2. Encoded categorical columns (Education, Marital_Status)
3. Removed 106 outliers using Isolation Forest (contamination=0.05)
4. Selected 16 high-variance features (top 75%)
5. Standardized features using StandardScaler

---

## Setup Instructions

### Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Get the Data
The dataset is included in `data/marketing_campaign.csv` (tab-separated format)

### Run the Notebook
```bash
cd mini-project-3
jupyter notebook notebooks/analysis.ipynb
```

Execute cells in order from top to bottom. Key sections:
1. **Imports & Data Loading** - Load and explore the dataset
2. **Data Preprocessing** - Clean, encode, filter outliers, select features
3. **Clustering** - Run K-means and Spectral Clustering
4. **Analysis** - Interpret cluster characteristics and anomalies
5. **Visualization** - PCA and t-SNE dimensionality reduction plots

---

## Results Summary

### Optimal K Selection
K was evaluated from 1 to 10 using both the elbow method and silhouette score.  
Both methods indicated that **K = 2** was the most stable and interpretable choice, with a peak silhouette score of **0.234**. While moderate, this level of separation is realistic for overlapping marketing behavior data.

### Customer Segments Identified

Two meaningful customer segments emerged:

**1. Premium High-Value Customers (60.1%)**
- Higher average income (~$63,882 vs $34,748)
- Significantly higher spending across all product categories
- 25% campaign response rate
- More catalog and store purchases
- Represent the primary revenue-driving group

**2. Low-Engagement Budget Customers (39.9%)**
- Lower income
- Minimal spending
- 0% campaign response rate
- Higher browsing activity but fewer purchases
- More price-sensitive and less engaged overall

### Dimensionality Reduction Insights
PCA showed separation primarily along the first principal component, largely influenced by income and spending variables.  
t-SNE visualizations showed even clearer grouping, suggesting some non-linear structure in customer behavior.

### Anomaly Detection Insights
Using Isolation Forest with 5% contamination:

- 106 customers were flagged as anomalous
- Anomalies showed:
  - 417% higher campaign response rates
  - 255% higher sweet product spending
  - 239% higher meat spending
  - 51% higher income
- 9.2% of Premium customers were labeled anomalous
- Only 0.2% of Budget customers were anomalous

This suggests anomalies are not noise, but extremely high-value customer behaviors that may represent VIP-level segments.

---

### Optimal K Selection
TO DO FOR LATER DO NOT FORGET

## Team Contributions

**Developers:** Ledja Halltari, Henry Chen

**Tasks done in this project:**
- Data preprocessing and outlier removal analysis (contamination experiments: 0.03, 0.05, 0.10)
- Feature selection using variance analysis
- Clustering implementation (K-means and Spectral Clustering)
- Anomaly characterization and categorization
- PCA and t-SNE visualizations
- Business label creation for clusters
- Full notebook development and documentation
