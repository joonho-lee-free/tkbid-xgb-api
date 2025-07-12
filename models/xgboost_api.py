from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import xgboost as xgb
import traceback

router = APIRouter()

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@router.api_route("/predict-xgb", methods=["POST", "OPTIONS"])
async def predict_xgb(request: PredictRequest):
    if not request.data or len(request.data) < 2:
        return {"error": "예측을 위한 충분한 데이터가 없습니다."}

    df = pd.DataFrame(request.data)

    try:
        df['target'] = df['낙찰가/기초가'].shift(-1)
        df = df.dropna(subset=['target'])

        if df.empty:
            return {"error": "예측 가능한 데이터가 없습니다."}

        # ✅ 피처 계산 (프론트에서 안 보내는 항목들)
        df['평균낙찰차이'] = df['평균낙찰가기초가'] - df['낙찰가/기초가']
        df['3회낙찰차이'] = df['평균3회낙찰가기초가'] - df['낙찰가/기초가']
        df['6회낙찰차이'] = df['평균6회낙찰가기초가'] - df['낙찰가/기초가']

        feature_cols = [
            '기초가격', '낙찰예정가격', '하한율', '낙찰가',
            '평균낙찰가기초가', '예상입찰가', '차액', '오차',
            '평균3회낙찰가기초가', '예상3입찰가', '차액3', '오차3',
            '평균6회낙찰가기초가', '예상6입찰가', '차액6', '오차6',
            '평균낙찰차이', '3회낙찰차이', '6회낙찰차이', 'winner'
        ]

        # ✅ 디버깅 로그
        print("🟢 마지막 3행 확인:\n", df.tail(3)[feature_cols].to_dict())
        print("🟢 feature_cols 존재 여부:", all(col in df.columns for col in feature_cols))

        X = df[feature_cols]
        y = df['target']

        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
        model.fit(X, y)

        last_row = df.iloc[[-1]][feature_cols]
        print("🟢 예측용 마지막 행:\n", last_row.to_dict())

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

    except Exception as e:
        return {
            "error": f"예측 중 오류 발생: {str(e)}",
            "trace": traceback.format_exc()
        }
