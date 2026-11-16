# EstateIQ — AI-powered Property Discovery & Price Prediction

An M.Tech CSE (AI & ML) academic project prototype combining supervised ML regression, a provider-switchable LLM assistant, retrieval-augmented generation (RAG), and an optional tabular GAN experiment.

> **Data honesty:** The included property records and locality rate seeds are synthetic demo data. This application does not scrape portals or access live listings. Do not present the sample listings, price estimates, or model scores as real market evidence. For a dissertation evaluation, replace the demo CSV with licensed, verified, time-stamped listing or transaction data.

## Architecture

- **Frontend:** Responsive HTML/CSS/JavaScript dashboard (no Node build required).
- **Backend:** FastAPI REST API.
- **ML:** Scikit-learn preprocessing pipeline + VotingRegressor (Random Forest 65%, HistGradientBoosting 35%).
- **LLM:** OpenAI, Gemini and Ollama adapter. Switch providers in Settings.
- **RAG:** ChromaDB persistent collection when installed; TF-IDF retrieval fallback for lightweight local runs.
- **GAN:** Optional PyTorch WGAN-GP script for a tabular synthetic-data augmentation experiment.
- **Storage:** CSV dataset, joblib model, ChromaDB local persistence.

## Quick start (Windows / VS Code)

1. Install Python 3.10 or newer.
2. Open this project folder in VS Code.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If PyTorch installation fails or is too large, skip it initially. It is only needed for the optional GAN experiment. ChromaDB is also optional; the app falls back to TF-IDF retrieval if ChromaDB cannot initialize.

5. Start the application from the project root:

   ```powershell
   uvicorn backend.main:app --reload
   ```

6. Open **http://127.0.0.1:8000**. API documentation is at **http://127.0.0.1:8000/docs**.

## LLM provider setup

Open **Settings & providers** in the web app.

- **Ollama:** Install and start Ollama, then run `ollama pull qwen2.5:7b`. Default endpoint: `http://localhost:11434`.
- **OpenAI:** Enter an API key and model name (default `gpt-4.1-mini`).
- **Gemini:** Enter a Gemini API key and model name (default `gemini-2.5-flash`). The adapter uses Gemini's OpenAI-compatible Chat Completions endpoint.

The provider selection and API key are stored in browser localStorage for this local prototype. This is not suitable for a public deployment. For production, move secrets to backend environment variables or a secrets manager, add authentication, rate limits, request validation, and HTTPS.

## Dataset format

Upload your own CSV through the API endpoint `POST /api/dataset/upload` (multipart form field: `file`) or replace `data/properties.csv`. Required columns:

`city, locality, property_type, bedrooms, bathrooms, area_sqft, age_years, floor, parking, furnishing, price_inr`

At least 30 rows are required by the upload endpoint; substantially more verified records are recommended. Example values:

- `property_type`: Apartment, Independent House, Villa
- `furnishing`: Unfurnished, Semi-furnished, Furnished
- `price_inr`: numeric INR transaction/listing value

The app retrains the model after upload. Keep an untouched holdout set for thesis evaluation; the demo endpoint currently uses a random train/test split for an initial prototype only.

## GAN experiment

With PyTorch installed, run:

```powershell
python backend/gan_augmentation.py
```

This writes `data/gan_augmented.csv`. Compare model results trained on (a) real data only and (b) real + generated rows. Report distribution plots, privacy/leakage checks, and performance on a real-only untouched test set. Do not mix generated rows into the final test set.

## Suggested M.Tech research experiments

1. Compare Linear Regression, Random Forest, Gradient Boosting, XGBoost (optional), and the ensemble.
2. Report MAE, RMSE, R², residual plots, and error by city/locality/property type.
3. Use locality-grouped and time-based splits to test generalization and reduce geographic/time leakage.
4. Compare model performance before and after GAN augmentation on a real-only holdout.
5. Evaluate RAG retrieval using a manually labelled question–passage set (Recall@k / MRR) and compare LLM responses with and without retrieved context.
6. Compare OpenAI, Gemini and Ollama on response quality, latency, token/cost estimates, and groundedness using the same question set.
7. Add explainability with permutation importance or SHAP and include fairness/error analysis across localities.

## API endpoints

- `GET /api/health`
- `GET /api/meta`
- `POST /api/search`
- `POST /api/predict`
- `GET /api/metrics`
- `POST /api/train`
- `POST /api/chat`
- `POST /api/dataset/upload`

## Notes

- The `For rent` and `Sold comparisons` selectors currently filter the demo discovery workflow but the sample records are not independently verified rental or sold transactions. Replace/extend the schema with `listing_status`, `listed_at`, `sold_at`, and `monthly_rent` for a research-ready implementation.
- The model's low/high range is a simple ±12% display band, not a calibrated prediction interval.
- For a strong thesis, cite your data source and collection date, document data cleaning, deduplication, missing-value handling, leakage prevention, and reproducibility.
