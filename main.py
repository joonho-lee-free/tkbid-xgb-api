from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.xgboost_api import router as xgb_router

app = FastAPI()

# ✅ CORS 허용 설정 – 정확한 origin만 명시
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://tkbid.vercel.app"],  # ← 정확한 프론트 도메인 명시
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ XGBoost 예측 API 라우터 등록
app.include_router(xgb_router)
