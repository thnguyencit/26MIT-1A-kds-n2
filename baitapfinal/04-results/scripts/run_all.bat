@echo off
REM Chạy lại TOÀN BỘ thực nghiệm trên Windows (tương đương run_all.sh).
REM   scripts\run_all.bat              -> đầy đủ (~20-30 phút)
REM   scripts\run_all.bat skip         -> bỏ bước thực nghiệm, dùng lại kết quả đã có
REM Cần bật venv trước, hoặc đặt biến PYTHON trỏ tới python.exe.
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
if "%PYTHON%"=="" set PYTHON=python
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set COLUMNS=120
set DATA=..\02-data
set RES=..\04-results
set MC=%DATA%\modcloth_final_data.json.gz
set RTR=%DATA%\renttherunway_final_data.json.gz
if not exist %RES%\logs mkdir %RES%\logs
if not exist %RES%\models mkdir %RES%\models

echo == 1. Moi truong va thong ke du lieu
%PYTHON% -m sizerec info > %RES%\logs\info.txt 2>&1
%PYTHON% -m sizerec data %MC% > %RES%\logs\data_modcloth.txt 2>&1
%PYTHON% -m sizerec data %RTR% > %RES%\logs\data_rtr.txt 2>&1

echo == 2. Kiem thu pipeline
%PYTHON% tests\test_smoke.py > %RES%\logs\smoke_test.txt 2>&1 || goto :err

if /i "%1"=="skip" goto :train
echo == 3. Thuc nghiem ModCloth (vai phut)
%PYTHON% -m sizerec experiment %MC% --protocols cold warm --seeds 1 2 3 4 5 6 7 8 9 10 --boot 200 -o %RES%\experiment\modcloth > %RES%\logs\experiment_modcloth.txt 2>&1 || goto :err
echo == 3. Thuc nghiem RentTheRunway (khoang 15 phut)
%PYTHON% -m sizerec experiment %RTR% --protocols cold warm --seeds 1 2 3 4 5 6 7 8 9 10 --boot 200 -o %RES%\experiment\rtr > %RES%\logs\experiment_rtr.txt 2>&1 || goto :err

:train
echo == 4. Huan luyen mo hinh de xuat
%PYTHON% -m sizerec train %MC% -o %RES%\models\modcloth.pt > %RES%\logs\train_modcloth.txt 2>&1 || goto :err
%PYTHON% -m sizerec train %RTR% -o %RES%\models\rtr.pt > %RES%\logs\train_rtr.txt 2>&1 || goto :err

echo == 5. Vi du goi y size
%PYTHON% -m sizerec recommend %RES%\models\rtr.pt --item 126335 --height 165cm --weight 57kg --bust-size 34b --body-type petite --age 22 --explain > %RES%\logs\demo_recommend.txt 2>&1
%PYTHON% -m sizerec recommend %RES%\models\rtr.pt --item 126335 --user 691468 --explain >> %RES%\logs\demo_recommend.txt 2>&1
%PYTHON% -m sizerec recommend %RES%\models\modcloth.pt --item 539980 --height 5ft6in --bra-size 34 --cup b --hips 38 --explain >> %RES%\logs\demo_recommend.txt 2>&1

echo == 6. Ve hinh va xuat Excel
%PYTHON% scripts\make_figures.py %RES% || goto :err
%PYTHON% scripts\make_figures_en.py %RES% || goto :err
%PYTHON% scripts\export_excel.py %RES% || goto :err
echo Xong. Ket qua trong %RES%  (log chi tiet trong %RES%\logs)
exit /b 0
:err
echo LOI: xem log trong %RES%\logs
exit /b 1
