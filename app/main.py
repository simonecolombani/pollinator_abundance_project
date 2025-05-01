from fastapi import FastAPI

from api.endpoints.calculate import CalculateHandler
from core.config import settings

app = FastAPI(**settings)
app.include_router(CalculateHandler().router, prefix="/api/v1")
