#!/bin/sh
# Compile Remastra.exe (lanceur Windows 64 bits) avec MinGW-w64.
set -e
cd "$(dirname "$0")"
x86_64-w64-mingw32-windres launcher.rc -O coff -o launcher_res.o
x86_64-w64-mingw32-gcc -O2 -s -municode -mwindows -o ../Remastra.exe launcher.c launcher_res.o -lshell32
rm -f launcher_res.o
echo "OK -> Remastra.exe"
