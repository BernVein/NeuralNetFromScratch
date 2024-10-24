@echo off
g++ -o neural neuralNetwork.cpp inputLayerImp.cpp hiddenLayerImp.cpp outputLayerImp.cpp
if errorlevel 1 exit /b  # Check for compilation errors
.\neural  # This will run the compiled executable

pause
