@echo off
call "F:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d F:\Downloads\electron\sss
cargo run
