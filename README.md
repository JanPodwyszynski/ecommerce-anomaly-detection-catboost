# 🛒 E-Commerce Catalogue Quality Challenge | Anomaly Detection Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange.svg)](https://catboost.ai/)
[![Metric](https://img.shields.io/badge/Validation%20ROC--AUC-0.9538-brightgreen.svg)]()
[![Validation](https://img.shields.io/badge/Validation-Seller--Disjoint%205--Fold-yellow.svg)]()

Production-ready Machine Learning pipeline detecting synthetic catalogue distortions, fraudulent listings, and data-entry errors in large-scale e-commerce marketplaces.

Developed as part of the **DSC PJATK Catalogue Quality Challenge** by **Team 8**.

---

## 📌 Problem Overview & Business Objective

Mass import of product catalogues from independent merchants regularly introduces corruptions: wrong category tags, falsified prices, incorrect parcel weights, or inflated bundle counts.

* **Target = 1 (Distorted):** A listing where exactly one core attribute (`declared_category`, `price_pln`, `shipping_weight_kg`, or `pack_size`) was synthetically altered.
* **Target = 0 (Valid):** A legitimate listing, which naturally contains missing values (`NaN`), typos, and mentions of *"free accessories"*.
* **Objective:** Produce a calibrated error ranking $P(\text{target} = 1)$ to prioritize automated Quality Assurance audits.
* **Evaluation Metrics:** **ROC-AUC** (ranking metric) and **PR-AUC** (addressing the ~17% anomaly imbalance).

---

## 🔬 Dataset & Validation Rigor

* **Dataset Scale:** 16,000 training records (`train.csv`), 10,000 test records (`test.csv`).
* **Seller-Disjoint Structure:**
  $$\text{sellers}_{\text{train}} \cap \text{sellers}_{\text{test}} = \emptyset$$
  The training partition features 800 unique sellers and the test partition contains 500 sellers (exactly 20 listings per merchant). Because merchants exhibit unique markdown patterns, title formats, and price rounding rules, random K-Fold validation causes catastrophic data leakage.
* **Cross-Validation Strategy:** Fully deterministic **5-Fold `GroupKFold` split strictly by `seller_id`** (SHA-256 seed 42) ensuring genuine out-of-sample merchant generalization.

---

## ⚙️ Modular Feature Engineering

The pipeline transforms raw attributes and free-form text through dedicated scikit-learn compatible modules:

1. **Category Alignment (`CategoryProbabilityExtractor`):**
   * TF-IDF vectorization on combined titles and descriptions feeding a Logistic Regression classifier.
   * Generates `declared_cat_prob` and a binary mismatch indicator `category_mismatch`.
2. **Bundle Size Estimation (`PackSizeProbabilityExtractor`):**
   * Free-gift filtering parser that eliminates misleading phrases (*"free accessory"*, *"free piece"*) prior to estimating core item counts.
   * Generates `declared_pack_size_prob` and `pack_size_mismatch`.
3. **Fine-Grained Item Identification (`product_type`):**
   * Extracted directly from description templates immediately preceding `"by [Brand]"`. Resolves high-variance broad categories into 90 narrow item types.
4. **Brand Recovery (`BrandImputer`):**
   * Reconstructs missing `brand` values from text against known entities with 1-character typo tolerance.
5. **Relational Unit Economics & Outlier Detection (`DataPreprocessor`):**
   * Effective quantity: $\text{pack}_{\text{eff}} = \max(1, \text{pack\_size})$
   * Unit metrics: $\text{unit\_price} = \frac{\text{price\_pln}}{\text{pack}_{\text{eff}}}$, $\text{unit\_weight} = \frac{\text{shipping\_weight\_kg}}{\text{pack}_{\text{eff}}}$
   * Price-to-weight ratio: $\text{price\_per\_weight} = \frac{\text{price\_pln}}{\text{shipping\_weight\_kg} + \epsilon}$
   * In-fold category $Z$-scores computed strictly on training partitions to prevent data leakage:
     $$Z_{\text{price}} = \frac{\log(1 + \text{unit\_price}) - \mu_{\text{cat}}}{\sigma_{\text{cat}} + \epsilon}$$

---

## 🏆 Model Configuration & Validation Results

Predictions are produced by **`CatBoostClassifier`** with native tokenization for text fields (`title`, `description`) and categorical encoding:

* `iterations`: 1500
* `learning_rate`: 0.03
* `depth`: 6
* `l2_leaf_reg`: 3
* `auto_class_weights`: `'Balanced'`

### Benchmark Results (5-Fold Seller-Disjoint CV)
| Metric | Baseline | Final Pipeline |
| :--- | :---: | :---: |
| **OOF ROC-AUC** | ~0.7620 | **0.9538** |
| **OOF PR-AUC** | ~0.4350 | **0.9035** |

---

## 🧪 Controlled Feature Ablation Studies

| Configuration | Feature Count | OOF ROC-AUC | $\Delta$ (vs Full) | Conclusion |
| :--- | :---: | :---: | :---: | :--- |
| **Full Reference Model** | **16** | **0.9538** | *baseline* | Best overall discrimination and stability. |
| **Exp A: Without `product_type`** | 15 | 0.9266 | **-0.0272** | Confirms sub-item extraction is vital to resolve category mismatches. |
| **Exp B: Without Unit Economics & Z-Scores** | 11 | 0.7345 | **-0.2193** | Absolute pricing and weight fail completely without pack-level normalization. |

---

## 🔍 Qualitative Error Analysis (OOF Audits)

* **False Positives ($p > 0.97$):**
  * `OFR_537731718`: Dual missing values combined with the high baseline corruption rate of the `music` category (36.6%) caused the model to over-penalize legitimate lightweight guitar strings.
  * `OFR_806144714`: A numeral typo (*"pack of fiv"*) bypassed the pack parser, treating a 5-item bundle as a single overpriced product.
  * `OFR_315719964`: Premium nail files (~19 PLN/unit) produced an anomalous $Z$-score against the general `beauty` baseline.
* **False Negatives ($p < 0.10$):**
  * `OFR_275857973`: Injected price distortion for a cooling pad fell well within the wide variance of the `computers` category.
  * `OFR_196691591`: Corrupted weight on a paperback travel guide matched the median weight for `books`.

---

## 👥 Authors & Contribution

Equal contribution project (50% / 50% split):

* **[Jan Podwyszyński](https://github.com/JanPodwyszynski):**  
  Full Exploratory Data Analysis (EDA) and distribution profiling; design of relational features (unit price/weight, price-to-weight ratio, category $Z$-scores); discovery and implementation of the `product_type` feature; numerical feature selection; execution of controlled ablation benchmarks; OOF error audit and final technical report.
* **[Michał Lichtarski](https://github.com/michal95pl):**  
  GitHub repository administration and Kaggle submission setup; design and implementation of modular NLP estimators in `features/` (`CategoryProbabilityExtractor`, `PackSizeProbabilityExtractor`, `BrandImputer`, `CrossValidationEnsemble`); initial baseline CatBoost deployment; code review and documentation.

---

## 📄 Documentation

The complete technical report is available in [`docs/team_8_report.pdf`](./docs/team_8_report.pdf).
