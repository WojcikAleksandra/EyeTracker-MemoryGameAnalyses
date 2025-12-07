@echo off
echo ========================================
echo Memory Game Eye Tracking Installation
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Testing installation...
python test_integration.py

echo.
echo To run the game:
echo   python MemoryGame_v2.py
echo.
pause

