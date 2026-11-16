from pathlib import Path
import sys, random
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from ml_engine import CITY_LOCALITIES, create_synthetic_dataset, train_model, predict
from llm_router import chat, DEFAULTS
from rag_engine import rag

app=FastAPI(title='EstateIQ API',version='1.0.0',description='M.Tech AI/ML property intelligence demo')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])

class PredictRequest(BaseModel):
 city:str='Hyderabad'; locality:str='Gachibowli'; property_type:str='Apartment'; bedrooms:int=2; bathrooms:int=2; area_sqft:int=1200; age_years:int=5; floor:int=3; parking:int=1; furnishing:str='Semi-furnished'
class SearchRequest(BaseModel):
 city:str='Hyderabad'; locality:str='Gachibowli'; intent:str='For sale'; min_bedrooms:int=1; max_budget:Optional[int]=None; property_type:str='Any'
class ChatRequest(BaseModel):
 question:str; provider:str='ollama'; model:str=''; api_key:str=''; base_url:str=''; history:list[dict]=Field(default_factory=list)

@app.on_event('startup')
def startup():
    if not (ROOT/'data'/'properties.csv').exists(): create_synthetic_dataset()
    train_model()

@app.get('/api/health')
def health(): return {'status':'ok','project':'EstateIQ','llm_providers':list(DEFAULTS)}
@app.get('/api/meta')
def meta(): return {'cities':{k:list(v.keys()) for k,v in CITY_LOCALITIES.items()},'providers':DEFAULTS}
@app.post('/api/search')
def search(req:SearchRequest):
    import pandas as pd
    path=ROOT/'data'/'properties.csv'; df=pd.read_csv(path)
    df=df[(df.city==req.city)&(df.locality==req.locality)&(df.bedrooms>=req.min_bedrooms)]
    if req.max_budget: df=df[df.price_inr<=req.max_budget]
    if req.property_type!='Any': df=df[df.property_type==req.property_type]
    # The demo CSV represents illustrative samples. Create curated, deterministic cards if a filter yields few rows.
    if len(df)<6:
        base=CITY_LOCALITIES.get(req.city,{}).get(req.locality,6500)
        rows=[]
        for i in range(9):
            beds=max(req.min_bedrooms, [1,2,2,3,3,4,2,3,4][i]); area=[650,980,1150,1450,1680,2200,1050,1550,2450][i]
            typ=req.property_type if req.property_type!='Any' else ['Apartment','Apartment','Apartment','Independent House','Apartment','Villa','Apartment','Independent House','Villa'][i]
            price=int(area*base*{'Apartment':1,'Independent House':1.13,'Villa':1.38}[typ]*random.Random(i+7).uniform(.9,1.13))
            rows.append({'city':req.city,'locality':req.locality,'property_type':typ,'bedrooms':beds,'bathrooms':min(5,beds),'area_sqft':area,'age_years':i*2,'price_inr':price,'intent':req.intent,'id':f'EQ-{i+101}'})
        df=pd.DataFrame(rows)
    df=df.copy(); df['intent']=req.intent; df['id']=[f'EQ-{i+101}' for i in range(len(df))]
    if req.intent=='For rent':
        # Demonstration-only implied monthly rent using a fixed 3.6% gross yield assumption.
        df['display_price']=(df.price_inr*0.036/12).round(-2).astype(int); price_label='per month'
    else:
        df['display_price']=df.price_inr.astype(int); price_label='sale price' if req.intent=='For sale' else 'indicative sold price'
    df['price_per_sqft']=(df.display_price/df.area_sqft).round().astype(int)
    if req.max_budget: df=df[df.display_price<=req.max_budget]
    median=int(df.display_price.median()) if len(df) else 0
    return {'city':req.city,'locality':req.locality,'intent':req.intent,'price_label':price_label,'count':len(df),'median_price':median,'avg_price':int(df.display_price.mean()) if len(df) else 0,'avg_rate':int(df.price_per_sqft.mean()) if len(df) else 0,'listings':df.head(18).to_dict(orient='records'),'data_note':'Synthetic demonstration listings — rent is inferred from an assumed gross yield, not a verified rent quote. Connect a verified property feed before real-world use.'}
@app.post('/api/predict')
def prediction(req:PredictRequest):
    try: return predict(req.model_dump())
    except Exception as e: raise HTTPException(400,str(e))
@app.post('/api/dataset/upload')
async def upload_dataset(file: UploadFile = File(...)):
    import pandas as pd
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(400,'Please upload a CSV file.')
    target=ROOT/'data'/'properties.csv'
    content=await file.read()
    try:
        import io
        df=pd.read_csv(io.BytesIO(content))
        required=set(['city','locality','property_type','bedrooms','bathrooms','area_sqft','age_years','floor','parking','furnishing','price_inr'])
        missing=required-set(df.columns)
        if missing: raise ValueError('Missing columns: '+', '.join(sorted(missing)))
        if len(df)<30: raise ValueError('Please provide at least 30 rows for a meaningful training experiment.')
        df[list(required)].to_csv(target,index=False)
        bundle=train_model(force=True)
        return {'status':'uploaded_and_trained','rows':len(df),'metrics':bundle['metrics']}
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/api/metrics')
def metrics(): return train_model()['metrics']
@app.post('/api/train')
def retrain():
    try:
        create_synthetic_dataset(); bundle=train_model(force=True); return {'status':'trained','metrics':bundle['metrics']}
    except Exception as e: raise HTTPException(500,str(e))
@app.post('/api/chat')
def assistant(req:ChatRequest):
    context=rag.retrieve(req.question,5)
    system={'role':'system','content':'You are EstateIQ, a careful real-estate research assistant. Use the retrieved context where relevant. Clearly distinguish synthetic demo data from verified market data. Never present estimates as guaranteed valuations or legal/financial advice. If the context is insufficient, say so.\nRETRIEVED CONTEXT:\n'+'\n'.join(context)}
    messages=[system]+req.history[-8:]+[{'role':'user','content':req.question}]
    try: answer=chat(req.provider,req.model,req.api_key,req.base_url,messages)
    except Exception as e: raise HTTPException(502,f'LLM provider error: {e}')
    return {'answer':answer,'sources':context,'provider':req.provider,'model':req.model or DEFAULTS.get(req.provider,{}).get('model','')}

app.mount('/',StaticFiles(directory=str(ROOT/'frontend'),html=True),name='frontend')
