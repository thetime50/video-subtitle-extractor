@echo off
call conda activate video-subtitle
cd /d d:\1024\python\video-subtitle-extractor
python .\backend\tools\repite_test.py %*
pause 