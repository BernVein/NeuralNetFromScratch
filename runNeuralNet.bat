@echo off
g++ -o neural neuralNetwork.cpp inputLayerImp.cpp hiddenLayerImp.cpp outputLayerImp.cpp
if errorlevel 1 exit /b  # Check for compilation errors
.\neural  # This will run the compiled executable

pause

@REM @echo off
@REM g++ -g -o neural "neuralNetwork.cpp" "inputLayerImp.cpp" "hiddenLayerImp.cpp" "outputLayerImp.cpp"
@REM if errorlevel 1 exit /b  # Check for compilation errors

@REM echo Compilation successful, launching GDB...
@REM gdb .\neural

@REM pause


