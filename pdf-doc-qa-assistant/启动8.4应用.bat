@echo off
chcp 65001 >nul
title 智能文档问答助手 (8.4)

REM 切换到本脚本所在目录（app.py / .env 都在这里）
cd /d "%~dp0"

REM 若 7860 端口已被占用，先释放（避免端口冲突启动失败）
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :7860 ^| findstr LISTENING') do (
    taskkill /F /PID %%p >nul 2>&1
)

echo ============================================
echo   智能文档问答助手 正在启动...
echo   启动后浏览器会自动打开 http://127.0.0.1:7860/
echo   关闭本窗口即可停止服务
echo ============================================
echo.

REM 用系统 Python 3.12 运行（官方 hello-agents 0.2.8 装在这里）
"C:\Users\CAITIANYU\AppData\Local\Programs\Python\Python312\python.exe" app.py

echo.
echo 服务已停止。
pause
