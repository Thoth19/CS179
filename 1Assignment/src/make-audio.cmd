@echo off
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" goto 64BIT
copy ..\AudioUtils\lib\lib32\libsndfile-1.dll
nvcc -L ..\AudioUtils\lib\lib32\ -I ..\AudioUtils\inc\ blur_host.cpp blur_device.cu libsndfile-1.lib -o blur
goto END
:64BIT
copy ..\AudioUtils\lib\lib64\libsndfile-1.dll
nvcc -L ..\AudioUtils\lib\lib64\ -I ..\AudioUtils\inc\ blur_host.cpp blur_device.cu libsndfile-1.lib -o blur
:END

del blur.lib
del blur.exp

