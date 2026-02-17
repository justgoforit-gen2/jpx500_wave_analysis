@echo off
REM JPX500 日次データ更新バッチ
REM タスクスケジューラで毎日 18:00 (市場閉場後) に実行推奨
REM
REM 設定方法:
REM   1. Win+R → taskschd.msc
REM   2. 「タスクの作成」→ トリガー: 毎日 18:00
REM   3. 操作: プログラムの開始
REM      プログラム: C:\Users\justg\Documents\python_projects\dify_projects\jpx500_wave_analysis\daily_update.bat
REM   4. 条件: 「コンピューターをAC電源で使用している場合のみ」のチェックを外す

cd /d "%~dp0"
echo [%date% %time%] Daily update started >> data\daily_update_history.log

.venv\Scripts\python.exe batch\update.py

echo [%date% %time%] Daily update finished (exit code: %ERRORLEVEL%) >> data\daily_update_history.log
