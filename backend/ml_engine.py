"""Trainable property-price ML pipeline. Demo data is synthetic and clearly labelled."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'properties.csv'
MODEL_PATH = ROOT / 'models' / 'price_model.joblib'
CITY_LOCALITIES = {
 'Hyderabad': {'Gachibowli': 7800, 'Madhapur': 9200, 'Kondapur': 7200, 'Kukatpally': 5600, 'Miyapur': 4900, 'Tellapur': 5200, 'Manikonda': 6100},
 'Kolkata': {'New Town': 5200, 'Salt Lake': 6900, 'Rajarhat': 4800, 'Tollygunge': 6100, 'Behala': 3900, 'Howrah': 3400, 'Dum Dum': 4100},
 'Bengaluru': {'Whitefield': 8500, 'Sarjapur Road': 7800, 'Electronic City': 5200, 'Hebbal': 9100, 'HSR Layout': 10500},
 'Pune': {'Hinjewadi': 5900, 'Wakad': 7200, 'Baner': 9000, 'Kharadi': 8500, 'Hadapsar': 6500},
 'Mumbai': {'Thane': 11500, 'Andheri': 23000, 'Powai': 26000, 'Navi Mumbai': 13500, 'Borivali': 17500},
 'Delhi': {'Dwarka': 9200, 'Rohini': 8500, 'Saket': 16500, 'Gurgaon': 12500, 'Noida': 7800},
}
FEATURES = ['city','locality','property_type','bedrooms','bathrooms','area_sqft','age_years','floor','parking','furnishing']

def create_synthetic_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    rows=[]
    for _ in range(n):
        city = rng.choice(list(CITY_LOCALITIES))
        locality = rng.choice(list(CITY_LOCALITIES[city]))
        base = CITY_LOCALITIES[city][locality]
        typ = rng.choice(['Apartment','Independent House','Villa'], p=[.72,.20,.08])
        beds = int(rng.choice([1,2,3,4,5], p=[.12,.38,.32,.14,.04]))
        area = int(np.clip(rng.normal(520 + beds*360, 170), 350, 3000))
        baths = int(np.clip(beds + rng.choice([-1,0,0,1]), 1, 5))
        age = int(rng.integers(0,25)); floor = int(rng.integers(0,25))
        parking = int(rng.choice([0,1,2], p=[.22,.68,.10]))
        furnishing = rng.choice(['Unfurnished','Semi-furnished','Furnished'], p=[.35,.45,.20])
        type_factor = {'Apartment':1.0,'Independent House':1.13,'Villa':1.38}[typ]
        furnishing_factor = {'Unfurnished':.97,'Semi-furnished':1.0,'Furnished':1.06}[furnishing]
        rate = base * type_factor * furnishing_factor * (1 - min(age,30)*.006) * rng.normal(1,.10)
        price = max(12_00_000, area * rate)
        rows.append([city,locality,typ,beds,baths,area,age,floor,parking,furnishing,round(price,-3)])
    df=pd.DataFrame(rows,columns=FEATURES+['price_inr'])
    DATA.parent.mkdir(exist_ok=True); df.to_csv(DATA,index=False)
    return df

def train_model(force=False):
    if MODEL_PATH.exists() and not force:
        return joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA) if DATA.exists() else create_synthetic_dataset()
    X=df[FEATURES]; y=df.price_inr
    cats=['city','locality','property_type','furnishing']; nums=[c for c in FEATURES if c not in cats]
    prep=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cats),('num',StandardScaler(),nums)],sparse_threshold=0.0)
    rf=RandomForestRegressor(n_estimators=220,min_samples_leaf=2,max_features=.9,random_state=42,n_jobs=-1)
    gb=HistGradientBoostingRegressor(max_iter=180,learning_rate=.07,l2_regularization=2,random_state=42)
    model=Pipeline([('prep',prep),('regressor',VotingRegressor([('rf',rf),('gb',gb)],weights=[.65,.35]))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    metrics={'mae':round(float(mean_absolute_error(yte,pred))), 'rmse':round(float(mean_squared_error(yte,pred)**.5)), 'r2':round(float(r2_score(yte,pred)),4), 'train_rows':len(Xtr),'test_rows':len(Xte),'dataset':'synthetic_demo'}
    MODEL_PATH.parent.mkdir(exist_ok=True); joblib.dump({'model':model,'metrics':metrics},MODEL_PATH)
    return {'model':model,'metrics':metrics}

def predict(payload):
    bundle=train_model(); x=pd.DataFrame([{k:payload[k] for k in FEATURES}])
    estimate=float(bundle['model'].predict(x)[0])
    return {'predicted_price':round(estimate/100000)*100000,'price_per_sqft':round(estimate/float(payload['area_sqft'])),'low_estimate':round(estimate*.88/100000)*100000,'high_estimate':round(estimate*1.12/100000)*100000,'metrics':bundle['metrics'],'disclaimer':'Illustrative model estimate trained on synthetic demonstration data; not a verified market valuation.'}
