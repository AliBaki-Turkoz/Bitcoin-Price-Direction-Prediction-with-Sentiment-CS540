# Bitcoin Price Direction Prediction Using News & Social-Media Sentiment

| | |
|---|---|
| **Course** | CS540 – Machine Learning in Finance, Özyeğin University (M.Sc. Computer Engineering), Fall 2025 · Supervisor: Emre Sefer |
| **Type** | Team project · repository maintained by Ali Baki Türköz |
| **Deliverables** | IEEE-style research paper (not included in this repository), presentation, notebook |
| **Headline result** | Next-day 3-class direction: best baseline macro-F1 **0.369 ± 0.062** (Price + News, Logistic Regression, 9-fold walk-forward). Chance level ≈ 0.33. |

## Problem
The project predicts the **direction of Bitcoin's next-day log return** as one of three classes: **Up**, **No change** or **Down**. It asks one question: do sentiment signals add predictive value beyond price features? The sentiment signals tested are:
- news sentiment,
- Twitter sentiment,
- the Crypto Fear & Greed Index.

## Data
| Dataset | Content | Size / evaluation |
|---|---|---|
| **Dataset 1** | BTC daily price + Crypto Fear & Greed Index + daily news sentiment | 709 days, regular · 9 walk-forward folds |
| **Dataset 2** | BTC daily price + daily news sentiment + daily Twitter sentiment | 167 days, irregular (only overlapping dates) · 3 folds |

Sources:
- [Crypto News](https://www.kaggle.com/datasets/oliviervha/crypto-news) (Kaggle, oliviervha)
- [Twitter BTC](https://www.kaggle.com/datasets/tylerdurden73/twitter-btc) (Kaggle)
- [Bitcoin Daily Price 2015–2025](https://www.kaggle.com/datasets/gamzegedik044/bitcoin-daily-price-20152025) (Kaggle)
- [Crypto Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/) (alternative.me)

Texts were scored with **VADER**, filtered to BTC-related items and averaged per day. The merged modelling tables are in `data/processed/`.

## Method
| Step | Details |
|---|---|
| Label | Next-day log return split by **tercile thresholds (q0.33 / q0.67)**. The thresholds are **recomputed in every fold from that fold's training window only**, which prevents look-ahead bias and keeps the classes balanced. |
| Evaluation | **Expanding-window walk-forward**. Results are reported as mean ± std over folds. The primary metric is macro-F1; accuracy, precision and recall are also reported. |
| Ablation | Feature sets: Price only · Price + Fear & Greed · Price + News · All (Dataset 1); Price · Price + News · Price + Twitter · All (Dataset 2) |
| Models | Logistic Regression, Linear SVM, RBF SVM, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, MLP (1–2 hidden layers, 4–8 units). Balanced sample weights are used for all models. |
| Tuning | Only the **top-5 (setup, model)** baseline combinations. Tuning used an inner time-series split on the first outer training window, and the tuned models were then re-evaluated with the same walk-forward protocol. |

## Results (macro-F1, mean ± std over folds)
| Dataset | Best baseline | Note |
|---|---|---|
| 1 (709 days) | **Price + News, Logistic Regression: 0.369 ± 0.062** | Price only, Linear SVM: 0.362 ± 0.049. Adding *both* Fear & Greed and News did **not** help. |
| 1, MLP after tuning | Price, MLP (1 hidden layer, 4 units): 0.383 ± 0.066 | modest gain |
| 2 (167 days) | Price only, Logistic Regression / Linear SVM: 0.336 ± 0.092 | Twitter sentiment tended to *lower* ML scores (e.g. 0.311). |
| 2, MLP | All signals, MLP (1 hidden layer, 4 units): 0.366 ± 0.028 baseline → 0.422 ± 0.138 after tuning | **unstable**: only 3 folds, large std |

**Interpretation**
- Sentiment helps in some periods and setups, but the effect is **conditional**: it depends on sample size, how regular the series is, and model capacity. More features did not reliably give better results.
- The "No change" class is easiest to identify. Up and Down are often confused with each other.
- Tuning **did not consistently help ML models**. On Dataset 1 all top-5 ML candidates dropped after tuning, which points to regime shifts after the early tuning window. Some MLP gains appeared, but they are not robust.
- Overall, all scores are close to chance (≈ 0.33). This is a realistic, leakage-safe evaluation of a hard problem, **not** a trading signal.

The full tables, confusion matrices and discussion are in the course paper (not included). The paper's confusion-matrix figures are in `report/`.

## Repository structure
```
notebooks/Crypto Price Prediction Using Sentiment Score.ipynb   # full pipeline: merging, VADER scoring, models, tuning (with outputs)
data/raw/          # daily price, Fear & Greed, daily VADER news/tweet scores
data/processed/    # merged modelling tables for Dataset 1 and Dataset 2 (.csv)
report/            # figures from the paper (table previews, confusion matrices)
presentation/      # project presentation (.pptx)
requirements.txt
```

## Running
1. Install the packages: `pip install -r requirements.txt`. The notebook was run on Google Colab in 2025; exact versions were not recorded.
2. Open the notebook with `notebooks/` as the working directory (Jupyter's default). Data paths are relative (`../data/...`), and the Colab-only Drive lines are commented out.
3. The modelling sections run directly from `data/processed/`. The data-preparation cells need the original raw dumps, which are **not included**: they are large and come from the Kaggle sources above. Expected paths:
   ```
   data/raw/News.csv · News_with_vader.csv · twitter_btc.csv · twitter_btc_with_vader.csv
   data/raw/Bitcoin_tweets_with_sentiment.csv · bitcoin_2017_to_2023_dakikalık.csv
   ```

## Notes
- **Working notebooks are not published.** The project was developed in separate working notebooks and then merged into the single notebook published here. Separately tried LSTM and other larger models stayed below the linear baselines (macro-F1 ≈ 0.15–0.32). They were not part of the paper and are not included.
- **The notebook is kept as run.** Its outputs show the original Colab environment, such as Drive mount messages.
- **Data licences:** the derived data tables come from public Kaggle datasets and the alternative.me index. Refer to the original sources for their terms.

## Academic integrity
This is graded coursework, shared as a portfolio project. It was carried out as a team; this repository does not claim it as individual work.

## License
The code in this repository is released under the [MIT License](LICENSE) (copyright the project team). Datasets, pretrained weights and third-party material keep their own licences and terms of use; they are not covered by this licence.
