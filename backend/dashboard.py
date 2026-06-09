from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()

HTML_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    if HTML_PATH.exists():
        return HTML_PATH.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
