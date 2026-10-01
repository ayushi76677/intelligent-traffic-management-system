$ErrorActionPreference = "Stop"

Write-Host "==============================================="
Write-Host " SMART TRAFFIC MANAGEMENT SYSTEM"
Write-Host " Production Server"
Write-Host "==============================================="

Set-Location $PSScriptRoot

Write-Host ""
Write-Host "Starting Waitress..."
Write-Host "URL: http://127.0.0.1:5000"
Write-Host ""

python -m waitress --host=127.0.0.1 --port=5000 --threads=4 app:app
