# 파일 위치: backend/main.py

from fastapi import FastAPI
from models.xgboost_api import router as predict_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 예측 라우터 연결
app.include_router(predict_router)
