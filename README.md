# Near-Duplicate Detection: Boilerplate-Aware LSH Pipeline

### Shiv Nadar University · Department of Computer Science  
**Authors:** Aastha Malhotra, Raashi Sharma, Sonali Verma  

---

## Overview

Duplicate and near-duplicate web pages significantly affect search engine performance by:
- Increasing storage requirements  
- Reducing indexing efficiency  
- Degrading search result quality  

This project explores a **structural solution** to improve near-duplicate detection by introducing a **boilerplate-aware preprocessing pipeline** before applying Locality Sensitive Hashing (LSH).

---

## Objective

To evaluate whether **removing HTML boilerplate (headers, footers, ads, navigation, etc.) before fingerprinting** improves:

- Precision  
- Recall  
- False positive rate  

Especially for **same-site web pages**, where structural similarity is high.

---

## Key Idea

Two pipelines are compared:

### Pipeline A — Raw DOM (Baseline)
- Direct tokenization from HTML  
- Includes boilerplate + content  

### Pipeline B — Extraction-First (Proposed)
- Uses `trafilatura` to extract main content  
- Removes structural noise before tokenization  

---

## Dataset

- **150 real web articles** from:
  - BBC News  
  - Al Jazeera  
  - Wikipedia  

- **900 labeled pairs** generated using controlled modifications  

### Variant Types

| Variant | Label | Description |
|--------|------|------------|
| timestamp_swap | 1 | Only date changed |
| ad_block_injection | 1 | Ads inserted |
| url_difference_only | 1 | URL modified |
| minor_content_edit | 1 | Small synonym changes |
| boilerplate_only_diff | 1 | Only HTML structure changed |
| same_site_diff_content | 0 | Same template, different content |

---

## Algorithms Implemented

### 1. MinHash + Supershingles
- Shingling (k = 8)  
- 84 hash functions  
- 6 supershingles  
- Threshold: ≥ 2 matches  

---

### 2. Charikar SimHash
- 384-bit fingerprint  
- Cosine similarity approximation  
- Threshold: ≤ 12 bit difference  

---

### 3. Hyperplane LSH (Extended SimHash)
- 512-bit fingerprints  
- Binary term frequency  
- Bigram augmentation  
- Improved discrimination  

---

### 4. MinHash LSH Banding
- 7 bands × 12 rows  
- Efficient candidate generation  
- Reduces O(n²) comparisons  

---

### 5. Combined Algorithm
- MinHash candidate generation  
- SimHash filtering  
- Improves precision  

---

## Experimental Setup

- Each algorithm tested on both pipelines  
- Metrics:
  - Precision  
  - Recall  
  - F1 Score  

- Additional analysis:
  - Precision-Recall curves  
  - Per-variant performance  
  - Cross-domain comparison  

---

## Key Hypotheses

- **H1:** Boilerplate removal improves detection of identical content  
- **H2:** Boilerplate removal reduces false positives  
- **H3:** Overall precision improves with extraction-first pipeline  

---

## Tech Stack

- Python 3  
- NumPy  
- BeautifulSoup (lxml parser)  
- Trafilatura (content extraction)  
- Datasketch (MinHash)  

---

## Project Structure
```
├── dataset/
│ ├── raw_html/ 
│ ├── extracted_text/ 
│ ├── articles.jsonl 
│ ├── pairs.jsonl 
│ ├── pairs_with_content.jsonl
│ └── stats.json 
│
├── hyperplane/
│ ├── hyperplane_detector.py 
│ ├── analyze_hyperplane.py 
│ └── analyze_pr_curve.py 
│
├── results/ 
│
├── dataset.py 
├── LSH_banding.py 
├── neardup_detector.py
├── requirements.txt 

```


## How to Run

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
pip install -r requirements.txt
python main.py
