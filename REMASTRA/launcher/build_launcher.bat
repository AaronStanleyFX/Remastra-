@echo off
rem Compile Remastra.exe avec MinGW-w64 (gcc + windres dans le PATH)
cd /d "%~dp0"
windres launcher.rc -O coff -o launcher_res.o
gcc -O2 -s -municode -mwindows -o ..\Remastra.exe launcher.c launcher_res.o -lshell32
del launcher_res.o
echo OK
