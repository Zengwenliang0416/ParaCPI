@echo off
set /p message="Enter commit message: "
set /p branch="Enter branch name (default is master): "

if "%branch%"=="" set branch=master

git add .
git commit -m "%message%"
git push origin %branch%

echo.
echo Commit and push have been executed.
pause
