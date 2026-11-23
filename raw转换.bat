@echo off
chcp 65001 >nul
cd /d "%~dp0"
setlocal enabledelayedexpansion

rem ============================================================
rem  用法: raw转换.bat [输入文件夹] [输出文件夹]
rem
rem    参数1 输入文件夹: 存放 *.RawPlain16LSB1 的目录
rem           默认: 本脚本所在文件夹\20261121_222944_image_dump
rem    参数2 输出文件夹: PNG 输出目录, 不存在会自动创建
rem           默认: 本脚本所在文件夹
rem
rem  相对路径均相对于"脚本所在文件夹"解析(脚本会先 cd 到这里)
rem  PNG 文件名 = 原 RAW 文件名, 输出目录里同名文件会被覆盖
rem
rem  ------------------------------------------------------------
rem  重要: 本文件必须保存为 CRLF(Windows) 换行 + UTF-8 无 BOM。
rem        若编辑器把它改成 LF 换行, cmd.exe 解析括号块时会错位,
rem        报 "'nv' 不是内部或外部命令" 这类莫名其妙的错误。
rem  依赖: uv 创建的 .venv
rem    uv venv .venv --python 3.12
rem    uv pip install --python .venv\Scripts\python.exe numpy opencv-python pillow
rem ============================================================

set "PY=%~dp0.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8

rem ---- 参数1: 输入文件夹 ----
set "SRC=%~1"
if "%SRC%"=="" set "SRC=%~dp020261121_222944_image_dump"
if "!SRC:~-1!"=="\" set "SRC=!SRC:~0,-1!"

rem ---- 参数2: 输出文件夹 ----
set "OUT=%~2"
if "%OUT%"=="" set "OUT=%~dp0"
if "!OUT:~-1!"=="\" set "OUT=!OUT:~0,-1!"

if not exist "%PY%" (
    echo [错误] 找不到 uv 虚拟环境: %PY%
    echo 请先执行:
    echo     uv venv .venv --python 3.12
    echo     uv pip install --python .venv\Scripts\python.exe numpy opencv-python pillow
    pause
    exit /b 1
)
if not exist "%SRC%\" (
    echo [错误] 找不到输入文件夹: %SRC%
    echo 用法: %~nx0 ^<输入文件夹^> ^[输出文件夹^]
    pause
    exit /b 1
)
dir /b /a-d "%SRC%\*.RawPlain16LSB1" >nul 2>&1
if errorlevel 1 (
    echo [提示] 输入文件夹里没有 *.RawPlain16LSB1 文件: %SRC%
    pause
    exit /b 0
)

if not exist "%OUT%\" (
    mkdir "%OUT%" 2>nul
    if not exist "%OUT%\" (
        echo [错误] 无法创建输出文件夹: %OUT%
        pause
        exit /b 1
    )
)

echo ========================================
echo   批量转换 RawPlain16LSB1  RAW -^> PNG
echo   输入: %SRC%
echo   输出: %OUT%
echo ========================================
echo.

set /a COUNT=0
for %%f in ("%SRC%\*.RawPlain16LSB1") do set /a COUNT+=1

set /a IDX=0
set /a OK=0
set /a FAIL=0

for %%f in ("%SRC%\*.RawPlain16LSB1") do (
    set /a IDX+=1
    echo [!IDX!/%COUNT%] 正在处理: %%~nxf
    "%PY%" "%~dp0raw_bayer_to_image.py" "%%f" ^
        -o "%OUT%\%%~nf.png" ^
        --w 3840 --h 3840 --stride 7680 --bayer GBRG ^
        --black 60 --ae 8 --shoulder 0.20 --contrast 0.75 --sat 2.0 --vibrance 1.0 --warm 1.04 ^
        --neutral 1200,2700,2600,3400
    if errorlevel 1 (
        set /a FAIL+=1
        echo    [失败] %%~nxf
    ) else (
        set /a OK+=1
    )
    echo.
)

echo ========================================
echo   全部完成！共 %COUNT% 个文件, 成功 %OK%, 失败 %FAIL%
echo   输出的 png 在: %OUT%
echo ========================================
pause
