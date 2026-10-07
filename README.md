# Wine Quality Screening App

Interactive Streamlit app built on the Wine Quality case study (Random Forest). Enter the laboratory profile of a
*Vinho Verde* wine and the app returns a model score and a screening decision (send to sensory panel or routine),
with an adjustable threshold that shows the workload-vs-coverage trade-off.

## Project structure

```
.
├── app.py            # Streamlit interface
├── core.py           # shared constants and helpers
├── train_model.py    # trains the model and saves model/wine_rf.joblib
├── data/             # winequality-red.csv, winequality-white.csv
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Run with Docker

```bash
docker build -t wine-quality-app .
docker run --rm -p 8501:8501 wine-quality-app
```

Open http://localhost:8501

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Notes

- Hyperparameters are set in `RF_PARAMS` inside `train_model.py` (taken from the notebook's tuned model).
- The model score is a ranking score, not a calibrated probability.
- The app shows statistical associations from a historical dataset; it does not replace sensory evaluation.

Data: Cortez et al. (2009), UCI Machine Learning Repository (CC BY 4.0).
