@echo off
call .\.venv\Scripts\activate
start python api.py
timeout /t 5 >nul
start streamlit run eval.py
start "" "index.html"
exit