"""Optional tabular GAN experiment for M.Tech documentation.

Trains a compact WGAN-GP on normalized numeric property attributes and writes
synthetic numeric rows. Categorical fields are sampled from observed frequencies.
Use only for augmentation experiments; never treat generated rows as market facts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parents[1]
NUM=['bedrooms','bathrooms','area_sqft','age_years','floor','parking','price_inr']

def run(epochs=300, n_generate=1000):
    torch.manual_seed(42); np.random.seed(42)
    source=ROOT/'data'/'properties.csv'; df=pd.read_csv(source)
    x=df[NUM].astype(float).to_numpy(); lo=x.min(0); hi=x.max(0); span=np.maximum(hi-lo,1)
    z=torch.tensor(((x-lo)/span*2-1),dtype=torch.float32)
    loader=DataLoader(TensorDataset(z),batch_size=128,shuffle=True,drop_last=True)
    G=nn.Sequential(nn.Linear(16,64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,len(NUM)),nn.Tanh())
    D=nn.Sequential(nn.Linear(len(NUM),64),nn.LeakyReLU(.2),nn.Linear(64,64),nn.LeakyReLU(.2),nn.Linear(64,1))
    go=torch.optim.Adam(G.parameters(),lr=1e-4,betas=(0,.9)); do=torch.optim.Adam(D.parameters(),lr=1e-4,betas=(0,.9))
    for epoch in range(epochs):
        for (real,) in loader:
            for _ in range(3):
                noise=torch.randn(len(real),16); fake=G(noise).detach(); alpha=torch.rand(len(real),1)
                mixed=(alpha*real+(1-alpha)*fake).requires_grad_(True); score=D(mixed)
                grad=torch.autograd.grad(score.sum(),mixed,create_graph=True)[0]
                gp=((grad.norm(2,dim=1)-1)**2).mean()
                loss_d=D(fake).mean()-D(real).mean()+10*gp
                do.zero_grad();loss_d.backward();do.step()
            loss_g=-D(G(torch.randn(len(real),16))).mean();go.zero_grad();loss_g.backward();go.step()
    with torch.no_grad(): samples=G(torch.randn(n_generate,16)).numpy()
    out=(samples+1)/2*span+lo; syn=pd.DataFrame(out,columns=NUM)
    syn['bedrooms']=syn.bedrooms.round().clip(1,6).astype(int);syn['bathrooms']=syn.bathrooms.round().clip(1,6).astype(int)
    syn['area_sqft']=syn.area_sqft.round().clip(250,5000).astype(int);syn['age_years']=syn.age_years.round().clip(0,60).astype(int)
    syn['floor']=syn.floor.round().clip(0,60).astype(int);syn['parking']=syn.parking.round().clip(0,4).astype(int);syn['price_inr']=syn.price_inr.clip(500000,100000000).round(-3)
    for col in ['city','locality','property_type','furnishing']:
        syn[col]=np.random.choice(df[col].unique(),size=n_generate,p=df[col].value_counts(normalize=True).reindex(df[col].unique()).values)
    syn=syn[['city','locality','property_type','bedrooms','bathrooms','area_sqft','age_years','floor','parking','furnishing','price_inr']]
    target=ROOT/'data'/'gan_augmented.csv';syn.to_csv(target,index=False);print(f'Wrote {len(syn)} GAN-generated rows to {target}')
if __name__=='__main__':run()
