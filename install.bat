REM Install portable git to .\git
powershell Invoke-WebRequest -Uri https://github.com/git-for-windows/git/releases/download/v2.39.0.windows.1/PortableGit-2.39.0-64-bit.7z.exe -OutFile git-installer.exe
git-installer.exe -o ./git -y
del git-installer.exe

REM Install portable python to .\python
powershell Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.8/python-3.10.8-embed-amd64.zip -OutFile python.zip
powershell Expand-Archive python.zip -DestinationPath .\python
echo python310.zip>.\python\python310._pth
echo .>>.\python\python310._pth
echo ..>>.\python\python310._pth
echo .\Lib\site-packages>>.\python\python310._pth
echo.>>.\python\python310._pth
del python.zip

REM Pull annotation tool repository on top of current directory
.\git\bin\git init . 
.\git\bin\git remote add origin https://github.com/smirkingface/annotation-tool
.\git\bin\git fetch origin main
.\git\bin\git checkout main -f

REM Install pip and install python requirements
powershell Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
.\python\python get-pip.py
del get-pip.py
.\python\python -m pip install -r requirements.txt --no-warn-script-location
