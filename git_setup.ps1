$git = "C:\Program Files\Git\cmd\git.exe"

& $git init
& $git config user.name "Flavius"
& $git config user.email "flavius@github.local"

& $git add .
& $git commit -m "Initial commit: AI Detector App for Thesis"

& $git branch -M main
& $git remote remove origin 2>$null
& $git remote add origin https://github.com/FlaviusFarca/licenta.git

Write-Host "Pushing to GitHub... Please check for a login popup!"
& $git push -u origin main
