@echo off
nvcc blur_host.cpp blur_device.cu -o blur
del blur.lib
del blur.exp
