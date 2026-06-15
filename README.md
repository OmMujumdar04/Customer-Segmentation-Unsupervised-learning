#  Customer Segmentation using RFM Analysis & Unsupervised Learning

An end-to-end unsupervised machine learning project that segments customers of a UK-based e-commerce retailer into meaningful groups using RFM Analysis, K-Means Clustering, DBSCAN, and PCA — turning raw transaction data into actionable business intelligence.

---

##  Problem Statement

Businesses need to understand their customers to make smart marketing decisions. Rather than treating all customers the same, segmentation allows companies to:

- Identify their most valuable customers
- Detect churned or at-risk customers
- Allocate marketing budgets efficiently
- Personalise campaigns for each customer group

This project builds a data-driven customer segmentation system entirely from raw transaction data — no predefined labels, no target variable. The machine finds the natural groupings on its own.

---

##  Dataset

- **Source:** [UCI Machine Learning Repository — Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Size:** 541,909 transactions across 8 columns
- **Period:** December 2010 — December 2011
- **Retailer:** UK-based online gift and homeware store

| Column | Description |
|---|---|
| InvoiceNo | Unique order identifier |
| StockCode | Product identifier |
| Description | Product name |
| Quantity | Units purchased (negative = return) |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Price per unit in GBP |
| CustomerID | Unique customer identifier |
| Country | Customer's country |

---

##  Project Workflow

```
Raw Transaction Data
        ↓
Data Cleaning & Validation
        ↓
RFM Feature Engineering
        ↓
Scaling (StandardScaler)
        ↓
Dimensionality Reduction (PCA)
        ↓
Find Optimal K (Elbow Method)
        ↓
K-Means Clustering
        ↓
DBSCAN Outlier Detection
        ↓
Cluster Interpretation & Visualisation
        ↓
Business Recommendations
```

---

##  Data Cleaning

Raw transaction data required significant cleaning before any analysis:

### Issues Identified & Resolved

| Issue | Rows Affected | Action Taken | Reason |
|---|---|---|---|
| Missing CustomerID | 135,080 | Dropped | Cannot assign transaction to any customer |
| Negative UnitPrice | ~100 | Dropped | Logically impossible — data entry errors |
| Negative Quantity | 8,905 | Kept | Represent product returns — valid signal |
| Missing Description | 1,454 | Kept | Not needed for RFM analysis |

### Key Decision — Keeping Negative Quantities

Returns (negative quantities) were intentionally retained because they correctly reduce a customer's Monetary value:

```
Purchase: +£500
Return:   -£500
Net:       £0  ← Customer spent nothing net → correctly reflected in Monetary
```

### TotalSpend Column

Created a new column to capture per-row transaction value:

```python
data['TotalSpend'] = data['Quantity'] * data['UnitPrice']
```

**After cleaning:** 406,829 transactions, 4,372 unique customers

---

##  RFM Feature Engineering

RFM is a proven framework used by retailers worldwide to measure customer value through three behavioural dimensions:

| Metric | Definition | Calculation |
|---|---|---|
| **Recency** | How recently did they buy? | Days since last purchase |
| **Frequency** | How often do they buy? | Count of unique orders |
| **Monetary** | How much do they spend? | Sum of all TotalSpend |

```python
reference_date = data['InvoiceDate'].max()

rfm = data.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', 'max'),
    Frequency = ('InvoiceNo',   'nunique'),
    Monetary  = ('TotalSpend',  'sum')
)

rfm['Recency'] = (reference_date - rfm['Recency']).dt.days
```

### Important Details

- **Reference date** = last date in the dataset (not today) since this is historical data
- **Frequency** uses `nunique()` on InvoiceNo — not row count — to count orders, not product lines
- **Negative Monetary customers** (returned more than purchased) were removed — 50 customers dropped

**Final RFM dataset:** 4,322 customers

---

## 🔧 Preprocessing

### Why Scaling Is Necessary

RFM features operate on completely different scales:

```
Recency   → 0 to 373 days
Frequency → 1 to 248 orders
Monetary  → £0 to £279,489
```

Without scaling, K-Means distance calculations would be dominated entirely by Monetary, ignoring Recency and Frequency. Every clustering decision would essentially just be "who spent the most."

### Why StandardScaler Over MinMaxScaler

MinMaxScaler compresses values between 0 and 1 using min and max. With extreme outliers (Monetary max = £279,489), all normal customers get squished near 0 — making the outlier problem worse.

StandardScaler centres data around mean=0 with std=1, making it more robust to outliers.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
```

### PCA — Dimensionality Reduction

Applied PCA to reduce 3 features to 2 principal components for visualisation:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)
```

**Variance retained:**
```
Component 1 → 55.9%
Component 2 → 30.1%
Total       → 86.0%  (above 80% threshold — acceptable)
```

Reducing to 2 dimensions allows clear 2D scatter plot visualisation of clusters — making results explainable to non-technical stakeholders.

---

##  Modelling

### Finding Optimal K — Elbow Method

Tested K from 1 to 10 and plotted inertia (total distance of points from their centroids):

```python
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(rfm_pca)
    inertias.append(kmeans.inertia_)
```

The elbow appeared at **K=3** — where the inertia curve stopped dropping aggressively. Beyond K=3, adding more clusters provided diminishing returns.

### K-Means Clustering

```python
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_pca)
```

Used `init='k-means++'` to intelligently spread initial centroids — reducing the instability caused by random initialisation.

### DBSCAN — Outlier Detection

```python
dbscan = DBSCAN(eps=1.5, min_samples=5)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_pca)
```

DBSCAN confirmed 11 genuine outliers (labelled `-1`) — customers so extreme in behaviour that they don't belong to any dense cluster. These are the ultra-high-value VIP buyers that K-Means had forced into Cluster 2.

---

##  Results

### Cluster Profiles

| Cluster | Size | Recency (days) | Frequency (orders) | Monetary (£) | Label |
|---|---|---|---|---|---|
| 0 | 3,220 | 241 | 1.86 | £487 |  Lost Customers |
| 1 | 1,088 | 38 | 5.78 | £1,953 |  Regular Customers |
| 2 | 14 | 6 | 105 | £106,628 |  VIP / Champions |

### Cluster Visualisation

K-Means produced three clearly separated segments in PCA space — confirming meaningful cluster boundaries with minimal overlap.

DBSCAN correctly identified the 14 VIP customers as genuine outliers — isolated far from the main customer population.

### K-Means vs DBSCAN Comparison

| Aspect | K-Means | DBSCAN |
|---|---|---|
| Clusters found | 3 meaningful segments | 1 cluster + noise |
| Outlier handling | Forces into nearest cluster | Labels as -1 (noise) |
| Business usability | High — actionable segments | Low for segmentation |
| Best use here |  Customer segmentation |  Outlier detection |

**Conclusion:** K-Means was better suited for segmentation on this dataset. DBSCAN complemented it by validating which customers are true outliers rather than a real segment.

---

##  Business Recommendations

### Cluster 0 — Lost Customers (74% of base)
- Last purchased **8 months ago** on average
- Low spend, low engagement
- **Action:** Win-back email campaign — "We miss you" with a significant discount code
- **Expected outcome:** Even 10-15% reactivation rate adds meaningful revenue

### Cluster 1 — Regular Customers (25% of base)
- Active within the **last 38 days**
- Moderate frequency and spend
- **Action:** Loyalty rewards programme, personalised product recommendations
- **Goal:** Push Regular customers toward VIP behaviour — increase order frequency
- **Priority:** Highest ROI — retaining existing customers is 5x cheaper than reacquiring lost ones

### Cluster 2 — VIP Champions (0.3% of base)
- Bought within the **last 6 days**
- Extremely high frequency and spend
- **Action:** Exclusive early access, premium membership, dedicated account management
- **Goal:** Protect and nurture — these 14 customers drive disproportionate revenue

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and RFM aggregation |
| NumPy | Numerical operations |
| Scikit-learn | StandardScaler, PCA, KMeans, DBSCAN |
| Matplotlib | Base plotting |
| Seaborn | Statistical visualisations |
| Jupyter Notebook | Development environment |

---

##  Key Learnings

- **Informative missingness** — missing CustomerIDs weren't random; they indicated untracked transactions that had to be removed entirely
- **Negative quantities are valid data** — returns reduce net Monetary correctly and should be retained
- **Scaling before clustering is non-negotiable** — without it, high-magnitude features dominate distance calculations
- **PCA is a preprocessing step, not just compression** — it enables visualisation and removes noise before clustering
- **Algorithm selection depends on purpose** — K-Means for segmentation, DBSCAN for outlier detection; neither is universally better
- **Business interpretation matters more than model accuracy** — unsupervised learning has no metric; the value is in the story the clusters tell

---
