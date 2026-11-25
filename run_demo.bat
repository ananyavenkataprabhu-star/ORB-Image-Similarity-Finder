
@echo off
cd /d "%~dp0"

echo ==============================
echo ORB IMAGE SIMILARITY FINDER
echo ==============================

call .venv\Scripts\activate.bat

echo.
echo STEP 1: IMAGE SIMILARITY SEARCH
python src/search.py dataset/test.jpg dataset --top 5 --method orb

pause

echo.
echo STEP 2: ORB KEYPOINT DETECTION
python src/orb_detector.py

pause
