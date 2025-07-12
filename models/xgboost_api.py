# 파일 위치: backend/models/xgboost_api.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import xgboost as xgb

router = APIRouter()

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@router.post("/predict-xgb")
async def predict_xgb(request: PredictRequest):
    df = pd.DataFrame(request.data)
    df['target'] = df['낙찰가/기초가'].shift(-1)
    df = df.dropna(subset=['target'])

    feature_cols = [
        '기초가격', '낙찰예정가격', '하한율', '낙찰가', '평균낙찰가기초가', '예상입찰가', '차액', '오차',
        '3회평균낙찰가기초가', '3회예상입찰가', '3회차액', '3회오차',
        '6회평균낙찰가기초가', '6회예상입찰가', '6회차액', '6회오차',
        '평균낙찰차이', '3회낙찰차이', '6회낙찰차이', 'winner'
    ]

    X = df[feature_cols]
    y = df['target']

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X, y)

    last_row = df.iloc[[-1]][feature_cols]
    predicted_ratio = model.predict(last_row)[0]
    base_price = df.iloc[-1]['기초가격']
    predicted_bid_price = round(base_price * predicted_ratio)

    importances = model.feature_importances_
    importance_dict = sorted(
        [
            {"name": name, "importance": round(float(score), 4)}
            for name, score in zip(feature_cols, importances)
        ],
        key=lambda x: x['importance'], reverse=True
    )[:5]

    return {
        "다음회차_예측_낙찰가_기초가": round(float(predicted_ratio), 5),
        "예상낙찰가": predicted_bid_price,
        "피처중요도": importance_dict
    }
