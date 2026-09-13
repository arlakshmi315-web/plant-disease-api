# Plant Disease Prediction API

FastAPI backend that classifies plant leaf diseases and returns treatment info.

## Files
- `app.py` - the API code
- `requirements.txt` - Python dependencies
- `plant_disease_model.h5` - trained model (add this yourself, see below)
- `class_indices.json` - class label mapping (add this yourself, see below)
- `disease_info.json` - disease/treatment database (add this yourself, see below)

## Deploy on Render
1. Push this folder to a GitHub repository (including the 3 files above).
2. On Render.com, create a New Web Service, connect this repo.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app:main --host 0.0.0.0 --port $PORT` -> use `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy. Render will give you a permanent URL like `https://plant-disease-api.onrender.com`

## Test
```
POST https://your-app.onrender.com/predict
Body: form-data, key "file", value: image file
```
