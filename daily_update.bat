@echo off
setlocal enabledelayedexpansion
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
set JPX_UPDATE_RC=!ERRORLEVEL!

echo [%date% %time%] Daily update finished (exit code: !JPX_UPDATE_RC!) >> data\daily_update_history.log

REM --- naibu-ryuho-app: jpx500_membership ユニバース同期 ---
REM 前提: jpx500 API (uvicorn api_server:app --port 8001) が起動中であること。
REM タスクスケジューラ等で常時起動推奨。
REM naibu は専用venvを持たず、jpx500側の .venv を共有して使用する運用。
if exist "..\naibu-ryuho-app\scripts\09_sync_jpx500_membership.py" (
    echo [%date% %time%] naibu sync started >> data\daily_update_history.log
    pushd "..\naibu-ryuho-app"
    "%~dp0.venv\Scripts\python.exe" scripts\09_sync_jpx500_membership.py
    set NAIBU_SYNC_RC=!ERRORLEVEL!
    popd
    echo [%date% %time%] naibu sync finished (exit code: !NAIBU_SYNC_RC!) >> "%~dp0data\daily_update_history.log"
) else (
    echo [%date% %time%] naibu-ryuho-app not found, skip sync >> data\daily_update_history.log
)

endlocal
