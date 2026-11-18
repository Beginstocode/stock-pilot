from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Src'/'data processing'))
import feature_engineering as fe
out=ROOT/'reports'/'targeted_vix_repair'; out.mkdir(parents=True,exist_ok=True)
prices=pd.read_csv(ROOT/'data/processed/stock_price_combined.csv')
prices=prices[prices.Symbol.astype(str).str.upper().isin(['RELIANCE','TCS'])].copy()
market=pd.read_csv(ROOT/'data/market/combined_market_data.csv')
source=pd.read_csv(ROOT/'data/raw/india_vix.csv',usecols=['Date','Close'])
source['Date']=fe._parse_dates(source.Date); source['Close']=pd.to_numeric(source.Close,errors='coerce'); source=source.dropna(subset=['Date']).drop_duplicates('Date',keep='first')
market['Date']=fe._parse_dates(market.Date)
market=market.drop_duplicates('Date').merge(source[['Date','Close']].rename(columns={'Close':'_source'}),on='Date',how='outer',validate='one_to_one')
market['India_VIX_Close']=pd.to_numeric(market.get('India_VIX_Close'),errors='coerce').combine_first(market['_source'])
market=market.drop(columns='_source').sort_values('Date').reset_index(drop=True)
market.to_csv(out/'combined_market_data.csv',index=False,date_format='%Y-%m-%d')
nifty=market[['Date','NIFTY50_Close']].rename(columns={'NIFTY50_Close':'Close'}).dropna()
vix=market[['Date','India_VIX_Close']].rename(columns={'India_VIX_Close':'Close'}).dropna()
features,_=fe.build_datasets(prices,nifty=nifty,vix=vix)
features.to_csv(out/'stock_features_RELIANCE_TCS.csv',index=False,date_format='%Y-%m-%d')
train=pd.read_csv(ROOT/'data/processed/training_dataset.csv',parse_dates=['Date']); inf=features
import json; cols=json.loads((ROOT/'reports/class_label_mapping.json').read_text())['feature_columns']
for date in ['2021-02-12','2021-03-30']:
 print('DATE',date,market.loc[market.Date.eq(pd.Timestamp(date)),['India_VIX_Close']].to_dict('records'))
for sym in ['RELIANCE','TCS']:
 a=train[train.Symbol.eq(sym)].set_index('Date'); b=inf[inf.Symbol.eq(sym)].set_index('Date'); d=a.index.intersection(b.index)
 x=fe.prepare_feature_matrix(a.loc[d],cols) if hasattr(fe,'prepare_feature_matrix') else None
 sys.path.insert(0,str(ROOT/'Src'/'modeling')); from feature_preprocessing import prepare_feature_matrix
 x=prepare_feature_matrix(a.loc[d],cols); y=prepare_feature_matrix(b.loc[d],cols)
 import numpy as np
 mask=~np.isclose(x.to_numpy(),y.to_numpy(),rtol=1e-6,atol=1e-6,equal_nan=True)
 print(sym,'mismatch_cells',int(mask.sum()))
