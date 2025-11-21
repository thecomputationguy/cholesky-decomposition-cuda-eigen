@echo off
REM --- Windows Batch File Equivalent of Linux Script ---

echo.
echo Cleaning old files...
del /q *.out
echo Done.
echo.
echo Compiling program...

REM --- NVCC Compilation Command for Windows ---
REM -std=c++17: Enforces C++17 standard (Required for modern C++ and Thrust/CCCL).
REM -O3: High optimization level.
REM -Xcompiler=/openmp: Passes the /openmp flag to the host compiler (MSVC).
REM -Xcompiler=/arch:AVX: Passes a high-performance architecture flag to MSVC.
REM -I C:\external_libraries\Eigen.
REM -lcusolver: Links the cuSOLVER library.
nvcc -std=c++17 -O3 -Xcompiler="/openmp" -Xcompiler="/arch:AVX" -I "C:\external_libraries" main.cu -lcusolver -o cholesky.exe

REM --- Error Check ---
if errorlevel 1 (
    echo.
    echo ERROR: Compilation failed!
    goto :eof
)

echo Done.

echo.
echo Setting OMP_NUM_THREADS and running...

REM --- Set Environment Variable (Windows syntax) ---
set OMP_NUM_THREADS=8

REM --- Execute the program (Windows executable name is cholesky.exe) ---
cholesky.exe

echo Done.

echo.
echo Plotting and saving graph.

REM --- Run Python Script (Using 'python' command) ---
python plot.py

echo Done.
echo.
pause