REM Stash local changes and update
.\git\bin\git stash
.\git\bin\git pull
.\git\bin\git stash pop

REM Run annotation tool
.\python\python main.py >log.txt 2>&1